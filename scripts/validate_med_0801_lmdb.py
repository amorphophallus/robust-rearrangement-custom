#!/usr/bin/env python3

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from src.dataset.lmdb import read_lmdb_episode_index, read_lmdb_meta


EXPECTED_TASK_COUNTS = {"one_leg": 200, "round_table": 200, "lamp": 200}
EXPECTED_EPISODES = sum(EXPECTED_TASK_COUNTS.values())
DEFAULT_SOURCE_SUFFIX = "rgbd-only-skill/med-rppo-base-0801"


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_pickle_identities(path: Path):
    identities = []
    for line in path.read_text().splitlines():
        _, _, relative_path = line.split("  ", 2)
        if relative_path.startswith("raw/"):
            relative_path = relative_path[len("raw/") :]
        identities.append(relative_path)
    return identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--annotation-mode", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-suffix", default=DEFAULT_SOURCE_SUFFIX)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    actual_manifest_sha256 = manifest_sha256(source_manifest)
    expected_pickle_files = manifest_pickle_identities(source_manifest)
    all_pickle_files = []
    all_pickle_paths = []
    combined_task_counts = Counter()

    for raw_path in args.paths:
        path = raw_path.expanduser().resolve()
        meta = read_lmdb_meta(path)
        attrs = meta["attrs"]
        episode_index = read_lmdb_episode_index(path)

        assert attrs["n_episodes"] == len(episode_index)
        assert attrs["selected_pickle_count"] == len(episode_index)
        shard_task_counts = Counter(item["task"] for item in episode_index)
        assert attrs["selected_task_counts"] == {
            task: shard_task_counts[task] for task in EXPECTED_TASK_COUNTS
        }
        assert attrs["tasks"] == ["one_leg", "round_table", "lamp"]
        assert attrs["randomness"] == "med"
        assert attrs["demo_source"] == "rollout"
        assert attrs["demo_outcome"] == "success"
        assert attrs["suffix"] == args.source_suffix
        assert attrs["output_suffix"] == args.suffix
        assert attrs["image_annotation_mode"] == args.annotation_mode

        pickle_files = attrs["pickle_files"]
        pickle_paths = attrs["pickle_paths"]
        assert len(pickle_files) == len(episode_index)
        assert len(pickle_paths) == len(episode_index)
        assert all(args.source_suffix in value for value in pickle_files)
        assert all("-smoke" not in value for value in pickle_files)
        assert all(int(item["success"]) == 1 for item in episode_index)
        assert [item["pickle_file"] for item in episode_index] == pickle_files

        provenance = attrs["provenance"]
        source_dataset = provenance["source_dataset"]
        assert source_dataset["episodes"] == EXPECTED_EPISODES
        assert source_dataset["task_counts"] == EXPECTED_TASK_COUNTS
        assert source_dataset["manifest_sha256"] == actual_manifest_sha256

        all_pickle_files.extend(pickle_files)
        all_pickle_paths.extend(pickle_paths)
        combined_task_counts.update(item["task"] for item in episode_index)

    assert len(all_pickle_files) == EXPECTED_EPISODES
    assert len(set(all_pickle_files)) == EXPECTED_EPISODES
    assert len(all_pickle_paths) == EXPECTED_EPISODES
    assert len(set(all_pickle_paths)) == EXPECTED_EPISODES
    assert set(all_pickle_files) == set(expected_pickle_files)
    assert combined_task_counts == Counter(EXPECTED_TASK_COUNTS), combined_task_counts

    print(
        f"[OK] paths={len(args.paths)}, episodes={EXPECTED_EPISODES}, "
        f"task_counts={EXPECTED_TASK_COUNTS}, suffix={args.suffix}, "
        f"annotation_mode={args.annotation_mode}, "
        f"source_manifest_sha256={actual_manifest_sha256}"
    )


if __name__ == "__main__":
    main()
