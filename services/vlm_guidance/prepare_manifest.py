"""Validate staged model files and write an immutable deployment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from services.vlm_guidance import POINT_POLICY_VERSION, SKILL_NAMES


REQUIRED_FILES = {
    "base_model": ("config.json",),
    "checkpoint": (
        "chat_template.jinja",
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


def _artifact_files(root_name: str, root: Path) -> list[str]:
    names = set(REQUIRED_FILES[root_name])
    if root_name == "base_model":
        weight_files = sorted(root.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(root.glob("pytorch_model*.bin"))
        if not weight_files:
            raise FileNotFoundError(f"no base-model weights found under {root}")
        names.update(path.name for path in weight_files)
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            if (root / index_name).is_file():
                names.add(index_name)
    return sorted(names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(args) -> dict:
    roots = {
        "base_model": Path(args.base_model_dir).resolve(),
        "checkpoint": Path(args.checkpoint_dir).resolve(),
    }
    with (roots["checkpoint"] / "config.json").open() as stream:
        policy = json.load(stream).get("hy_furniture_policy", {})
    if policy.get("version") != POINT_POLICY_VERSION:
        raise RuntimeError(f"unexpected point policy: {policy}")
    if tuple(policy.get("skill_names", ())) != SKILL_NAMES:
        raise RuntimeError("unexpected skill order")
    files = {}
    for root_name, root in roots.items():
        for name in _artifact_files(root_name, root):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files[f"{root_name}/{name}"] = {
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_revision": args.base_revision,
        "checkpoint_revision": args.checkpoint_revision,
        "point_policy_version": POINT_POLICY_VERSION,
        "skill_names": list(SKILL_NAMES),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_manifest(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
