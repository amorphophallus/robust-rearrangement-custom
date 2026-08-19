#!/usr/bin/env python3

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "med-rppo-base-0801"
TASKS = ("one_leg", "round_table", "lamp")
TARGET_PER_TASK = 200
PROVENANCE_PATH = ROOT / "logs/med_0801_provenance.json"
MANIFEST_PATH = ROOT / "logs/med_0801_source_manifest.sha256"
COMPLETE_PATH = ROOT / "logs/.med_0801_provenance_finalized"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(task: str):
    directory = (
        ROOT
        / "raw/raw/diffik/sim"
        / task
        / "rollout/med/rgbd-only-skill"
        / RUN_NAME
        / "success"
    )
    paths = sorted(directory.glob("*.pkl"))
    if len(paths) != TARGET_PER_TASK:
        raise RuntimeError(
            f"{task}: found {len(paths)} source pickles in {directory}; "
            f"expected {TARGET_PER_TASK}"
        )
    return paths


def main():
    entries = []
    task_counts = {}
    total_bytes = 0

    for task in TASKS:
        paths = source_paths(task)
        task_counts[task] = len(paths)
        for index, path in enumerate(paths, start=1):
            relative_path = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            entries.append((digest, size, relative_path))
            total_bytes += size
            print(
                f"[{task} {index:03d}/{len(paths)}] {digest} "
                f"{size} {relative_path}",
                flush=True,
            )

    manifest_text = "".join(
        f"{digest}  {size}  {relative_path}\n"
        for digest, size, relative_path in entries
    )
    unique_digests = {digest for digest, _, _ in entries}
    if len(unique_digests) != len(entries):
        raise RuntimeError(
            f"Source dataset contains exact duplicate pickle payloads: "
            f"{len(entries)} files but only {len(unique_digests)} unique SHA256 values"
        )
    manifest_tmp = MANIFEST_PATH.with_suffix(".sha256.tmp")
    manifest_tmp.write_text(manifest_text)
    manifest_tmp.replace(MANIFEST_PATH)
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    provenance = json.loads(PROVENANCE_PATH.read_text())
    provenance["source_dataset"] = {
        "finalized_at": datetime.now().astimezone().isoformat(),
        "run_name": RUN_NAME,
        "episodes": len(entries),
        "task_counts": task_counts,
        "total_bytes": total_bytes,
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": manifest_sha256,
        "manifest_format": "sha256  bytes  repository-relative-path",
    }
    provenance_tmp = PROVENANCE_PATH.with_suffix(".json.tmp")
    provenance_tmp.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance_tmp.replace(PROVENANCE_PATH)
    COMPLETE_PATH.write_text(manifest_sha256 + "\n")
    print(
        f"Finalized {len(entries)} source pickles, {total_bytes} bytes; "
        f"manifest_sha256={manifest_sha256}"
    )


if __name__ == "__main__":
    main()
