import json

import pytest

from src.dataset.base import EpisodeRef
from src.dataset.storage import apply_episode_selection_index


def _ref(index, task, source):
    return EpisodeRef(
        path_idx=0,
        episode_idx=index,
        frame_start=index * 10,
        frame_end=(index + 1) * 10,
        frame_count=10,
        task=task,
        success=1,
        domain="sim",
        source=source,
    )


def _write_index(path, include_tasks):
    payload = {
        "schema": "rr-episode-selection-v1",
        "selection_id": "test-selection",
        "rules": [
            {
                "source": "AutoMate",
                "include_tasks": include_tasks,
                "exclude_tasks": ["00003"],
                "expected_input_task_count": 3,
                "expected_input_episode_count": 6,
                "expected_output_task_count": 2,
                "expected_output_episode_count": 4,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_episode_selection_keeps_other_sources_and_filters_tasks(tmp_path):
    manifest = [
        _ref(0, "00001", "AutoMate"),
        _ref(1, "00001", "AutoMate"),
        _ref(2, "00002", "AutoMate"),
        _ref(3, "00002", "AutoMate"),
        _ref(4, "00003", "AutoMate"),
        _ref(5, "00003", "AutoMate"),
        _ref(6, "lamp", "FurnitureBench"),
    ]
    index_path = tmp_path / "index.json"
    _write_index(index_path, ["00001", "00002"])

    selected = apply_episode_selection_index(manifest, index_path)

    assert [ref.task for ref in selected if ref.source == "AutoMate"] == [
        "00001",
        "00001",
        "00002",
        "00002",
    ]
    assert [ref.task for ref in selected if ref.source == "FurnitureBench"] == [
        "lamp"
    ]


def test_episode_selection_rejects_missing_tasks(tmp_path):
    manifest = [
        _ref(0, "00001", "AutoMate"),
        _ref(1, "00002", "AutoMate"),
        _ref(2, "00003", "AutoMate"),
        _ref(3, "00003", "AutoMate"),
        _ref(4, "00003", "AutoMate"),
        _ref(5, "00003", "AutoMate"),
    ]
    index_path = tmp_path / "index.json"
    _write_index(index_path, ["00001", "00999"])

    with pytest.raises(ValueError, match="missing tasks"):
        apply_episode_selection_index(manifest, index_path)
