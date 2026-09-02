import json
from pathlib import Path

import numpy as np
import pytest

from src.data_processing.process_pickles_to_lmdb import (
    IMAGE_KEYS,
    LOWDIM_KEYS,
    combine_processed_episode_group,
    load_episode_groups_manifest,
)


def _episode_part(name: str, length: int, offset: int = 0):
    part = {
        "episode_length": length,
        "task": "one_leg",
        "success": 1,
        "env": None,
        "pickle_file": name,
    }
    for key in (*IMAGE_KEYS, *LOWDIM_KEYS):
        part[key] = np.arange(offset, offset + length, dtype=np.float32)[:, None]
    return part


def test_episode_group_manifest_resolves_portable_basenames(tmp_path: Path):
    selected = [tmp_path / "a.aligned-s00.pkl", tmp_path / "a.aligned-s01.pkl"]
    for path in selected:
        path.touch()
    manifest = tmp_path / "groups.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "test_groups_v1",
                "selective_episode_groups": [
                    {
                        "source": "a",
                        "segments": [0, 1],
                        "pickle_files": [
                            "/different/host/a.aligned-s00.pkl",
                            "/different/host/a.aligned-s01.pkl",
                        ],
                    }
                ],
            }
        )
    )

    groups, payload = load_episode_groups_manifest(manifest, selected)

    assert payload["schema"] == "test_groups_v1"
    assert len(groups) == 1
    assert groups[0]["paths"] == [path.resolve() for path in selected]
    assert groups[0]["segments"] == [0, 1]


def test_episode_group_manifest_requires_exact_coverage(tmp_path: Path):
    selected = [tmp_path / "a.pkl", tmp_path / "b.pkl"]
    for path in selected:
        path.touch()
    manifest = tmp_path / "groups.json"
    manifest.write_text(
        json.dumps(
            {
                "episode_groups": [
                    {"group_id": "only-a", "pickle_files": ["a.pkl"]}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="cover every selected pickle"):
        load_episode_groups_manifest(manifest, selected)


def test_combine_processed_episode_group_concatenates_all_timeseries():
    group = {
        "group_id": "source.stitched-g000",
        "source": "source",
        "segments": [3, 4],
    }

    combined = combine_processed_episode_group(
        group,
        [_episode_part("s03.pkl", 2, 0), _episode_part("s04.pkl", 3, 2)],
    )

    assert combined["episode_length"] == 5
    assert combined["pickle_file"] == "source.stitched-g000"
    assert combined["source_pickle_files"] == ["s03.pkl", "s04.pkl"]
    assert combined["segments"] == [3, 4]
    for key in (*IMAGE_KEYS, *LOWDIM_KEYS):
        np.testing.assert_array_equal(
            combined[key].reshape(-1), np.arange(5, dtype=np.float32)
        )


def test_combine_processed_episode_group_rejects_mixed_task():
    left = _episode_part("left.pkl", 1)
    right = _episode_part("right.pkl", 1)
    right["task"] = "lamp"

    with pytest.raises(ValueError, match="inconsistent task"):
        combine_processed_episode_group(
            {"group_id": "bad", "source": "source", "segments": []},
            [left, right],
        )
