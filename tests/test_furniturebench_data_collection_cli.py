import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_processing.process_pickles import process_pickle_file
from scripts.data_collection.collect_furniturebench import evaluation_environment


ROOT = Path(__file__).resolve().parents[1]


class FurnitureBenchDataCollectionCliTest(unittest.TestCase):
    def test_collection_subprocess_uses_requested_data_root(self):
        requested = Path("/tmp/furniturebench-production-root")
        self.assertEqual(evaluation_environment(requested)["DATA_DIR_RAW"], str(requested))

    def test_collect_dry_run_records_metadata_without_mutating_images(self):
        command = [
            sys.executable,
            str(ROOT / "scripts/data_collection/collect_furniturebench.py"),
            "--tasks",
            "one_leg",
            "--target-successes",
            "1",
            "--annotation-source",
            "scripted",
            "--output-suffix",
            "unit-test-clean-raw",
            "--base-seed",
            "20901000",
            "--dry-run",
        ]
        result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        output = result.stdout
        self.assertIn("--annotate-skill", output)
        self.assertIn("--enable-annotation-verify", output)
        self.assertIn("--annotation-source scripted", output)
        self.assertIn("--seed 20901000", output)
        self.assertIn("simulator_seed=20901000", output)
        self.assertIn("saved_image_annotation_mode=none", output)
        self.assertNotIn("--guidance-point-on-image", output)
        self.assertNotIn("--grasp-annotation-on-image", output)
        self.assertNotIn("--grasp-part-annotate", output)
        self.assertNotIn("--skill-on-image", output)

    def test_lmdb_dry_run_requires_explicit_offline_annotation(self):
        command = [
            sys.executable,
            str(ROOT / "scripts/data_collection/process_furniturebench_pickles_to_lmdb.py"),
            "--tasks",
            "one_leg",
            "--input-suffix",
            "clean-raw",
            "--output-suffix",
            "annotated-lmdb",
            "--image-annotation-mode",
            "guidance-point",
            "--episodes-per-task",
            "1",
            "--dry-run",
        ]
        result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        self.assertIn("source_pickle_image_annotation_mode=none", result.stdout)
        self.assertIn("lmdb_image_annotation_mode=guidance-point", result.stdout)
        self.assertIn("--image-annotation-mode guidance-point", result.stdout)
        self.assertIn("--require-source-image-annotation-mode none", result.stdout)

    def test_lmdb_processing_rejects_collection_annotated_source(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "contaminated.pkl"
            with path.open("wb") as stream:
                pickle.dump({"image_annotation_mode": "guidance-point"}, stream)
            with self.assertRaisesRegex(ValueError, "expected source image_annotation_mode='none'"):
                process_pickle_file(
                    path,
                    noop_threshold=0.0,
                    required_source_image_annotation_mode="none",
                )


if __name__ == "__main__":
    unittest.main()
