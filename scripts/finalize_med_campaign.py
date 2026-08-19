#!/usr/bin/env python3

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


TASKS = ("one_leg", "round_table", "lamp")
TARGET = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-suffix", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.manifest.resolve()
    provenance_path = args.provenance.resolve()
    marker = args.marker.resolve()
    entries = []
    task_counts = {}
    total_bytes = 0

    for task in TASKS:
        source_dir = (
            root / "raw/raw/diffik/sim" / task / "rollout/med"
            / args.source_suffix / args.run_name / "success"
        )
        paths = sorted(source_dir.glob("*.pkl"))
        if len(paths) != TARGET:
            raise RuntimeError(f"{task}: found {len(paths)} pickles, expected {TARGET}: {source_dir}")
        task_counts[task] = len(paths)
        for path in paths:
            digest = sha256_file(path)
            size = path.stat().st_size
            entries.append((digest, size, path.relative_to(root).as_posix()))
            total_bytes += size

    if len({digest for digest, _, _ in entries}) != len(entries):
        raise RuntimeError("source campaign contains duplicate pickle payloads")

    manifest_text = "".join(
        f"{digest}  {size}  {relative_path}\n"
        for digest, size, relative_path in entries
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest.with_name(manifest.name + f".tmp.{__import__('os').getpid()}")
    manifest_tmp.write_text(manifest_text)
    manifest_tmp.replace(manifest)
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()

    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance["source_dataset"] = {
        "finalized_at": datetime.now().astimezone().isoformat(),
        "run_name": args.run_name,
        "source_suffix": args.source_suffix,
        "episodes": len(entries),
        "task_counts": task_counts,
        "total_bytes": total_bytes,
        "manifest_path": str(manifest.relative_to(root)),
        "manifest_sha256": manifest_sha,
        "manifest_format": "sha256  bytes  repository-relative-path",
        "annotation_stage": "source_rollout_skill_metadata;image_annotation=pickle_to_lmdb",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(manifest_sha + "\n")
    print(f"Finalized {len(entries)} pickles, {total_bytes} bytes, manifest_sha256={manifest_sha}")


if __name__ == "__main__":
    main()
