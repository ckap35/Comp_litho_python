import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from imaging import ImagingModel


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_imports_do_not_run_the_example(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)
            process = subprocess.run(
                [sys.executable, "-c", "import run_mask_optimization; import Run_MO_CTM_class_based"],
                cwd=directory, env=environment, capture_output=True, text=True, check=True,
            )
            self.assertEqual(process.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_original_command_saves_a_consistent_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            process = subprocess.run(
                [sys.executable, str(ROOT / "Run_MO_CTM_class_based.py"),
                 "--iterations", "2", "--output", str(output)],
                env=environment, capture_output=True, text=True, check=True,
            )
            metrics = json.loads(process.stdout)
            self.assertEqual(metrics, json.loads((output / "metrics.json").read_text()))
            self.assertEqual(metrics["accepted_iterations"], 2)
            self.assertTrue(metrics["monotone_accepted_loss"])
            self.assertTrue(metrics["final_aerial_matches_recorded_loss"])
            with np.load(output / "arrays.npz") as arrays:
                np.testing.assert_allclose(ImagingModel().forward(arrays["mask"]), arrays["aerial"], atol=1e-14)
                self.assertAlmostEqual(np.sum((arrays["target"] - arrays["aerial"]) ** 2), metrics["final_loss"])
            self.assertTrue((output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue((output / "parameters.json").exists())


if __name__ == "__main__":
    unittest.main()
