import json
from types import SimpleNamespace

from services.vlm_guidance.prepare_manifest import build_manifest


def test_manifest_validates_policy_and_hashes_both_weight_sets(tmp_path):
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    base.mkdir()
    checkpoint.mkdir()

    (base / "config.json").write_text("{}")
    (base / "model-00001-of-00001.safetensors").write_bytes(b"base weights")

    checkpoint_config = {
        "hy_furniture_policy": {
            "version": 2,
            "skill_names": ["push", "pick", "place", "insert", "screw"],
        }
    }
    (checkpoint / "config.json").write_text(json.dumps(checkpoint_config))
    for name in (
        "chat_template.jinja",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (checkpoint / name).write_bytes(name.encode())

    manifest = build_manifest(
        SimpleNamespace(
            base_model_dir=str(base),
            checkpoint_dir=str(checkpoint),
            model_mode="structured",
            base_revision="base-revision",
            checkpoint_revision="checkpoint-revision",
        )
    )

    assert manifest["base_revision"] == "base-revision"
    assert manifest["checkpoint_revision"] == "checkpoint-revision"
    assert "base_model/model-00001-of-00001.safetensors" in manifest["files"]
    assert "checkpoint/model.safetensors" in manifest["files"]
    assert all(len(value["sha256"]) == 64 for value in manifest["files"].values())


def test_manifest_accepts_original_sft_without_base_model(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    )
    for name in (
        "chat_template.jinja",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (checkpoint / name).write_bytes(name.encode())

    manifest = build_manifest(
        SimpleNamespace(
            base_model_dir=None,
            checkpoint_dir=str(checkpoint),
            model_mode="original_sft",
            base_revision=None,
            checkpoint_revision="native-revision",
        )
    )

    assert manifest["model_mode"] == "original_sft"
    assert manifest["policy_version"] == 3
    assert set(manifest["files"]) == {
        f"checkpoint/{name}"
        for name in (
            "chat_template.jinja",
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
