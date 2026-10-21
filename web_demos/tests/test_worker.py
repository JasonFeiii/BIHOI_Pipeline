from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2
    import numpy as np
    import trimesh
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from worker import pose_overlay, read_wapr_inputs


@unittest.skipUnless(AVAILABLE, "requires numpy, opencv-python and trimesh in a model environment")
class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        p = Path(self.temp.name)
        cv2.imwrite(str(p / "rgb.png"), np.zeros((80, 100, 3), np.uint8))
        np.save(p / "depth.npy", np.full((80, 100), 1000, np.uint16))
        mask = np.zeros((80, 100), np.uint8)
        mask[20:60, 25:75] = 255
        cv2.imwrite(str(p / "mask.png"), mask)
        trimesh.creation.box(extents=(100, 100, 100)).export(p / "mesh.ply")
        self.config = {"rgb":str(p / "rgb.png"), "depth":str(p / "depth.npy"),
                       "mask":str(p / "mask.png"), "mesh":str(p / "mesh.ply"),
                       "bbox":None, "depth_scale":.001, "mesh_scale":.001,
                       "fx":100, "fy":100, "cx":50, "cy":40}

    def tearDown(self):
        self.temp.cleanup()

    def test_metric_conversion_preserves_original_resolution(self):
        rgb, depth, mesh, mask, box, K = read_wapr_inputs(self.config)
        self.assertEqual(rgb.shape, (80, 100, 3))
        np.testing.assert_allclose(depth, 1)
        np.testing.assert_allclose(mesh.extents, .1)
        self.assertEqual(np.count_nonzero(mask), 2000)
        self.assertIsNone(box)
        np.testing.assert_allclose(K, [[100,0,50],[0,100,40],[0,0,1]])

    def test_depth_size_mismatch_is_rejected(self):
        np.save(self.config["depth"], np.ones((40, 50)))
        with self.assertRaisesRegex(ValueError, "尺寸"):
            read_wapr_inputs(self.config)

    def test_empty_mask_and_invalid_depth_are_rejected(self):
        cv2.imwrite(self.config["mask"], np.zeros((80,100),np.uint8))
        with self.assertRaisesRegex(ValueError, "Mask 为空"):
            read_wapr_inputs(self.config)
        np.save(self.config["depth"], np.full((80,100),np.nan))
        with self.assertRaisesRegex(ValueError, "有效深度"):
            read_wapr_inputs({**self.config,"bbox":[10,10,20,20]})

    def test_box_outside_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "超出"):
            read_wapr_inputs({**self.config, "bbox":[90,70,20,20]})

    def test_overlay_projects_object_in_camera_coordinates(self):
        rgb, _, mesh, _, _, K = read_wapr_inputs(self.config)
        pose = np.eye(4)
        pose[2,3] = 1
        overlay = pose_overlay(rgb, mesh, pose, K, .1)
        self.assertEqual(overlay.shape, rgb.shape)
        self.assertGreater(np.count_nonzero(overlay[30:51,40:70]), 0)
        self.assertEqual(np.count_nonzero(overlay[:10,:10]), 0)


if __name__ == "__main__":
    unittest.main()
