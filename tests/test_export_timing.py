"""Exercise the real export resampler with synthetic motion; no weights required."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'pipeline/FoundationPose/export_standardized_mano_object_table_4rings_first500.py'
spec=importlib.util.spec_from_file_location('standard_export',SCRIPT)
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)

class ExportTimingTests(unittest.TestCase):
    def test_ten_to_thirty_fps_preserves_motion_duration(self):
        source=np.arange(10,41)
        poses=np.tile(np.eye(4),(len(source),1,1))
        poses[:,0,3]=(source-source[0])/10
        frames,coordinates,timestamps,hands,output=export.resample_all(source,10,30,{},poses)
        self.assertEqual(len(frames),91)
        self.assertAlmostEqual(timestamps[-1],3)
        np.testing.assert_allclose(coordinates,10+timestamps*10)
        np.testing.assert_allclose(output[:,0,3],timestamps)
        old=export.resample_all(source,30,30,{},poses)
        self.assertAlmostEqual(old[2][-1],1)

    def test_invalid_rates_are_rejected(self):
        for fps in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):export.resample_all([0,1],fps,30,{},np.tile(np.eye(4),(2,1,1)))
            with self.assertRaises(ValueError):export.resample_all([0,1],10,fps,{},np.tile(np.eye(4),(2,1,1)))

    def test_cli_requires_input_fps(self):
        result=subprocess.run([sys.executable,str(SCRIPT),'--data_name','example_sequence'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--input_fps',result.stderr)

if __name__=='__main__':unittest.main()
