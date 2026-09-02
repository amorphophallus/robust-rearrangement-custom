import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset.lmdb import validate_lmdb_metadata_contract


class LmdbMetadataContractTest(unittest.TestCase):
    def test_accepts_shared_and_domain_specific_requirements(self):
        metadata = {
            "attrs": {
                "domain": "real",
                "annotation_source": "scripted",
                "timeline_frequency_hz": 10,
                "contains_v6_offline_buffered": True,
            }
        }
        with patch("src.dataset.lmdb.read_lmdb_meta", return_value=metadata):
            result = validate_lmdb_metadata_contract(
                [Path("real.lmdb")],
                required_attrs={
                    "annotation_source": "scripted",
                    "timeline_frequency_hz": 10.0,
                },
                required_attrs_by_domain={
                    "real": {"contains_v6_offline_buffered": True}
                },
            )
        self.assertEqual(result["real.lmdb"]["domain"], "real")

    def test_rejects_wrong_annotation_mode_before_training(self):
        metadata = {"attrs": {"domain": "real", "image_annotation_mode": "none"}}
        with patch("src.dataset.lmdb.read_lmdb_meta", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "image_annotation_mode"):
                validate_lmdb_metadata_contract(
                    Path("real.lmdb"),
                    required_attrs={
                        "image_annotation_mode": "guidance-point-colored"
                    },
                )


if __name__ == "__main__":
    unittest.main()
