"""Regressions for export timing, server path boundaries and credential logging."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'webui'),str(ROOT/'scripts')]
import interaction_capture_webui as app
import settings
import download_models

class ReleaseSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'data';self.root.mkdir()
        self.env=patch.dict(os.environ,{'HOI_DATA_ROOT':str(self.root),'HOI_ALLOWED_DATA_ROOTS':'','HOI_ALLOWED_EXPORT_ROOTS':''})
        self.env.start();self.addCleanup(self.env.stop)

    def test_root_allowlist_and_symlink_escape(self):
        self.assertEqual(app._data_root_path(str(self.root)),self.root)
        outside=self.root.parent/'outside';outside.mkdir()
        for path in [outside,self.root.parent]:
            with self.assertRaises(ValueError):app._data_root_path(str(path))
        (self.root/'escape').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):app._data_root_path(str(self.root/'escape'))
        (self.root/'rigid').mkdir();(self.root/'rigid/person').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):app._safe_dataset_base(str(self.root),'person','sequence')

    def test_export_isolated_and_preserves_unrelated_files(self):
        data=self.root/'rigid/person/sequence'
        standard=data/app.EXPORT_STANDARD_DIR;standard.mkdir(parents=True)
        (standard/'metadata.json').write_text('{}')
        (data/'outputmano').mkdir();(data/'outputmano/video.mp4').write_bytes(b'fixture')
        destination=self.root/'exports/review';destination.mkdir(parents=True)
        unrelated=destination/'keep.txt';unrelated.write_text('keep')
        with patch.object(app,'_active_runs',return_value=[]):
            status,archive=app.export_datasets(str(self.root),'person',['sequence'],str(destination),True)
        self.assertIn('Exported 1',status)
        self.assertEqual(unrelated.read_text(),'keep')
        with zipfile.ZipFile(archive) as package:
            self.assertEqual(set(package.namelist()),{'sequence/video.mp4','sequence/standardized_hoi_30fps/metadata.json'})
        with self.assertRaises(ValueError):app._resolve_export_dir(str(self.root),'person',str(data))
        with self.assertRaises(ValueError):app._resolve_export_dir(str(self.root),'person',str(self.root.parent))

    def test_export_rejects_source_and_destination_symlink_escapes(self):
        data=self.root/'rigid/person/sequence';data.mkdir(parents=True)
        outside=self.root.parent/'outside';outside.mkdir()
        (outside/'private.mp4').write_bytes(b'private')
        (data/'outputmano').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):app._export_targets(data)
        (self.root/'exports').mkdir();(self.root/'exports/escape').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):app._resolve_export_dir(str(self.root),'person','exports/escape')

    def test_password_is_not_written_to_logs(self):
        log=io.StringIO()
        with patch.object(app,'RUNS_DIR',self.root/'runs'),contextlib.redirect_stdout(log):
            self.assertEqual(app._login_password('configured-secret'),('configured-secret',None))
            password,path=app._login_password(None)
        self.assertNotIn('configured-secret',log.getvalue());self.assertNotIn(password,log.getvalue())
        self.assertEqual(path.read_text().strip(),password)
        if os.name=='posix':self.assertEqual(path.stat().st_mode & 0o777,0o600)
        launch=MagicMock()
        log=io.StringIO()
        with patch.object(app,'RUNS_DIR',self.root/'runs'),patch.object(sys,'argv',['webui','--data-root',str(self.root),'--auth-password','configured-secret']),patch.object(app,'make_ui',return_value=launch),contextlib.redirect_stdout(log):
            app.main()
        self.assertNotIn('configured-secret',log.getvalue())
        self.assertEqual(launch.launch.call_args.kwargs['auth'][1],'configured-secret')
        self.assertIn(str(path.resolve()),launch.launch.call_args.kwargs['blocked_paths'])

    def test_final_export_receives_processed_frame_rate(self):
        with patch.object(app,'_ensure_pipeline_data_name',return_value='rigid/person/sequence'):
            steps=app._final_steps(app.Job('sequence','person','object'),app.EASYMOCAP_ROOT,self.root,500,10)
        command=next(step.command for step in steps if step.title=='Export standardized interaction data')
        args=shlex.split(command)
        self.assertEqual(float(args[args.index('--input_fps')+1]),10)
        self.assertEqual(float(args[args.index('--output_fps')+1]),30)

    def test_model_downloader_hash_check_and_no_overwrite(self):
        target=self.root/'model.pth'
        import hashlib
        contents=b'public model fixture';digest=hashlib.sha256(contents).hexdigest()
        with patch.object(download_models.urllib.request,'urlopen',return_value=io.BytesIO(contents)):
            download_models.download_verified('https://example.invalid/model',target,digest)
        self.assertEqual(target.read_bytes(),contents)
        with self.assertRaises(RuntimeError):download_models.download_verified('https://example.invalid/model',target,'0'*64)
        self.assertEqual(target.read_bytes(),contents)

if __name__=='__main__':unittest.main()
