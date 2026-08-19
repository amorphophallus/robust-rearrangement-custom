#!/usr/bin/env python3

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from src.dataset.lmdb import (
    EPISODE_DATA_PREFIX,
    META_KEY,
    episode_data_key,
    json_loads_bytes,
    open_lmdb_env,
    read_lmdb_episode_index,
    read_lmdb_meta,
    unpack_named_arrays,
)


EXPECTED_TASK_COUNTS = {"one_leg": 200, "round_table": 200, "lamp": 200}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-suffix", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-mode", default="none")
    parser.add_argument("--require-rollout-stage", action="store_true")
    args = parser.parse_args()

    path = args.path.resolve()
    meta = read_lmdb_meta(path)
    attrs = meta["attrs"]
    index = read_lmdb_episode_index(path)
    counts = Counter(item["task"] for item in index)
    assert len(index) == 600, len(index)
    assert counts == Counter(EXPECTED_TASK_COUNTS), counts
    assert attrs["n_episodes"] == 600
    assert attrs["selected_task_counts"] == EXPECTED_TASK_COUNTS
    assert attrs["suffix"] == f"{args.source_suffix}/{args.run_name}", attrs["suffix"]
    assert attrs["image_annotation_mode"] == args.annotation_mode
    assert attrs["demo_source"] == "rollout"
    assert attrs["randomness"] == "med"
    assert attrs["demo_outcome"] == "success"
    assert "skill" in meta["lowdim_specs"]

    manifest = args.manifest.resolve()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    provenance = attrs.get("provenance", {}).get("source_dataset", {})
    assert provenance.get("manifest_sha256") == manifest_sha, provenance
    if args.require_rollout_stage:
        assert provenance.get("annotation_stage") == "rollout", provenance

    skill_nonzero = 0
    env = open_lmdb_env(path, readonly=True)
    try:
        with env.begin(write=False) as txn:
            for episode_idx in (0, 200, 400):
                raw = txn.get(episode_data_key(episode_idx))
                assert raw is not None
                arrays = unpack_named_arrays(raw)
                skill_nonzero += int((arrays["skill"] != 0).any())
    finally:
        env.close()
    assert skill_nonzero == 3, skill_nonzero
    print(
        f"[OK] {path}: episodes=600 task_counts={dict(counts)} "
        f"skill_samples={skill_nonzero}/3 manifest_sha256={manifest_sha}"
    )


if __name__ == "__main__":
    main()
