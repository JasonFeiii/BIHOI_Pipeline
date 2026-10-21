import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import DemoState, Handler, ThreadingHTTPServer


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = DemoState(self.temp.name, interpreters={app: sys.executable for app in ("wapr",)})
        self.raw = dict(app="wapr", region="mask", backend="torch", fx=500, fy=500,
                        cx=320, cy=240, depth_scale=.001, mesh_scale=.001, diameter_m=.1)
        for key, suffix in (("rgb", ".png"), ("depth", ".npy"), ("mesh", ".ply"), ("mask", ".png")):
            token = str(len(self.raw)).zfill(32) + suffix
            (self.state.uploads / token).touch()
            self.raw[key] = token

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def test_parameter_validation(self):
        for field, bad in (("fx", 0), ("fy", "nan"), ("depth_scale", -1),
                           ("diameter_m", True), ("backend", "unknown"), ("device", "cpu")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.state.validate({**self.raw, field: bad})
        self.assertEqual(self.state.validate(self.raw)["fx"], 500)


    def test_reject_path_traversal_and_wrong_file_types(self):
        for token in ("../../server.py", "/etc/passwd", "b" * 32 + ".png", None):
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.state.upload_path(token)
        token = "c" * 32 + ".ply"
        (self.state.uploads / token).touch()
        with self.assertRaisesRegex(ValueError, "RGB"):
            self.state.validate({**self.raw, "rgb": token})

    def test_client_cannot_override_interpreter_or_checkpoint(self):
        result = self.state.validate({**self.raw, "python": "/tmp/untrusted",
                                      "checkpoint": "/tmp/untrusted", "repo": "/tmp/untrusted"})
        self.assertNotIn("python", result)
        self.assertEqual(result["repo"], self.state.repos["wapr"])
        self.assertNotIn("checkpoint", result)

    def test_disabled_app(self):
        self.state.app = "disabled"
        with self.assertRaises(ValueError):
            self.state.validate(self.raw)


    def test_worker_success_and_artifact_collection(self):
        class Process:
            def __init__(self, command, **kwargs):
                config = json.loads(Path(command[-1]).read_text())
                output = Path(config["output"])
                (output / "result.json").write_text('{"model":"WAPR"}')
                (output / "overlay.png").write_bytes(b"test artifact")
                self.stdout = io.StringIO("model loaded\ninference finished\n")

            def wait(self):
                return 0

        with patch("server.subprocess.Popen", Process):
            submitted = self.state.submit(self.raw)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                job = self.state.snapshot(submitted["id"])
                if job["status"] in ("succeeded", "failed"):
                    break
                time.sleep(.01)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"]["model"], "WAPR")
        self.assertIn("overlay.png", job["artifacts"])
        self.assertIn("results.zip", job["artifacts"])
        self.assertFalse((self.state.jobs_dir / job["id"] / "results.zip").exists())
        self.assertNotIn("input.json", job["artifacts"])
        self.assertNotIn("process", job)
        self.assertIn("inference finished", job["log"])

    def test_invalid_python_fails_with_actionable_message(self):
        self.state.python["wapr"] = "/does/not/exist/python"
        submitted = self.state.submit(self.raw)
        for _ in range(100):
            job = self.state.snapshot(submitted["id"])
            if job["status"] == "failed":
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "failed")
        self.assertIn("--wapr-python", job["log"])

    def test_cancel_queued_job_does_not_start_worker(self):
        self.state.jobs["queued"] = dict(id="queued", status="queued", process=None, log="")
        self.assertEqual(self.state.cancel("queued")["status"], "cancelled")
        with patch("server.subprocess.Popen") as popen:
            self.state.run("queued", {"output": self.temp.name, "app": "wapr"})
        popen.assert_not_called()

    def test_restart_restores_completed_job_without_eager_zip(self):
        directory = self.state.jobs_dir / ("b" * 32)
        directory.mkdir()
        (directory / "result.json").write_text('{"model":"WAPR"}')
        (directory / "overlay.png").write_bytes(b"video")
        restored = DemoState(self.temp.name)
        try:
            job = restored.snapshot(directory.name)
            self.assertEqual(job["status"], "succeeded")
            self.assertIn("results.zip", job["artifacts"])
            self.assertFalse((directory / "results.zip").exists())
        finally:
            restored.close()


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.state = DemoState(cls.temp.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.state = cls.state
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()
        cls.state.close()
        cls.temp.cleanup()

    def fetch(self, path, data=None, headers=None):
        return urlopen(Request(self.url + path, data=data, headers=headers or {}), timeout=5)

    def test_demo_page_and_assets(self):
        for path, content in (("/wapr", b"POSE LAB"), ("/", b"POSE LAB"),
                              ("/app.js", b"waprFields"), ("/style.css", b"@media")):
            with self.subTest(path=path), self.fetch(path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(content, response.read())
        with self.fetch("/api/config") as response:
            self.assertEqual(json.load(response)["app"], "wapr")

    def test_binary_upload(self):
        payload = bytes(range(256))
        with self.fetch("/api/upload", payload, {"X-Filename": "input.png", "Content-Type": "application/octet-stream"}) as response:
            result = json.load(response)
        self.assertEqual((self.state.uploads / result["token"]).read_bytes(), payload)
        self.assertEqual(result["size"], 256)

    def test_cross_origin_and_invalid_upload_are_rejected(self):
        for headers, expected in (({"X-Filename":"bad.py"}, 400),
                                  ({"X-Filename":"file.png", "Origin":"https://untrusted.example"}, 403)):
            with self.subTest(headers=headers), self.assertRaises(HTTPError) as raised:
                self.fetch("/api/upload", b"data", headers)
            self.assertEqual(raised.exception.code, expected)
            raised.exception.close()

    def test_result_video_range_and_private_config(self):
        folder = self.state.jobs_dir / ("f" * 32)
        folder.mkdir(exist_ok=True)
        (folder / "overlay.png").write_bytes(b"0123456789")
        (folder / "input.json").write_text("private")
        with self.fetch(f"/files/{folder.name}/overlay.png", headers={"Range":"bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        with self.assertRaises(HTTPError) as raised:
            self.fetch(f"/files/{folder.name}/input.json")
        self.assertEqual(raised.exception.code, 404)
        raised.exception.close()

    def test_zip_is_built_on_download_and_cached_without_recompressing_binary(self):
        folder = self.state.jobs_dir / ("e" * 32)
        folder.mkdir()
        (folder / "overlay.png").write_bytes(b"encoded image")
        (folder / "pose.npy").write_bytes(b"pose matrix")
        (folder / "result.json").write_text('{"model":"WAPR"}')
        (folder / "input.json").write_text("private")
        self.state.jobs[folder.name] = dict(id=folder.name, status="succeeded", process=None,
                                          log="", result={})
        self.assertFalse((folder / "results.zip").exists())
        with self.fetch(f"/files/{folder.name}/results.zip") as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                self.assertNotIn("input.json", archive.namelist())
                self.assertEqual(archive.read("overlay.png"), b"encoded image")
                self.assertEqual(archive.getinfo("overlay.png").compress_type, zipfile.ZIP_STORED)
                self.assertEqual(archive.getinfo("pose.npy").compress_type, zipfile.ZIP_DEFLATED)
        previous = (folder / "results.zip").stat().st_mtime_ns
        self.state.archive(folder.name)
        self.assertEqual((folder / "results.zip").stat().st_mtime_ns, previous)
        (folder / "result.json").write_text('{"model":"WAPR", "updated":true}')
        self.state.archive(folder.name)
        with zipfile.ZipFile(folder / "results.zip") as archive:
            self.assertTrue(json.loads(archive.read("result.json"))["updated"])


if __name__ == "__main__":
    unittest.main()
