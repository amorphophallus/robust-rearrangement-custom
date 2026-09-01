import argparse
import json
import pickle

import numpy as np
import pytest

from src import vlm_data_generator as generator


def _record(sample_id: str, assistant_payload: dict) -> dict:
    return {
        "id": sample_id,
        "image": [f"images/{sample_id}_front.png", f"images/{sample_id}_wrist.png"],
        "state_info": {},
        "messages": [
            {"role": "user", "content": "<image>\n<image>"},
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload),
            },
        ],
        "metadata": {"task": "one_leg"},
    }


def _valid_payload() -> dict:
    return {
        "skill": "screw",
        "target_point_2d": [233.0, 141.0],
        "target_point_3d": [0.334311, 0.137487, 0.504977],
    }


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"skill": None}, "skill_null"),
        ({"skill": "unknown"}, "skill_invalid"),
        ({"target_point_2d": None}, "target_point_2d_null"),
        ({"target_point_2d": [1.0]}, "target_point_2d_shape"),
        ({"target_point_2d": [1.0, float("nan")]}, "target_point_2d_non_finite"),
        ({"target_point_3d": None}, "target_point_3d_null"),
        ({"target_point_3d": [1.0, 2.0]}, "target_point_3d_shape"),
    ],
)
def test_supervision_error_rejects_invalid_labels(update, expected):
    payload = _valid_payload()
    payload.update(update)
    assert generator._supervision_error(payload) == expected


def test_supervision_error_checks_front_image_bounds():
    payload = _valid_payload()
    payload["target_point_2d"] = [320.0, 120.0]
    assert (
        generator._supervision_error(payload, front_image_shape=(240, 320, 3))
        == "target_point_2d_out_of_frame"
    )


def test_write_records_is_strict_about_null_supervision(tmp_path):
    payload = _valid_payload()
    payload["target_point_2d"] = None
    with pytest.raises(ValueError, match="target_point_2d_null"):
        generator._write_records(
            output_dir=tmp_path,
            records=[_record("bad", payload)],
            formats="messages-jsonl",
            llamafactory_state_mode="base",
        )
    assert not (tmp_path / "messages.jsonl").exists()


def test_to_llamafactory_skips_null_supervision(tmp_path):
    valid = _valid_payload()
    invalid = _valid_payload()
    invalid["target_point_2d"] = None
    input_file = tmp_path / "sharegpt.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "id": "valid",
                    "image": ["front.png", "wrist.png"],
                    "state_info": {},
                    "conversations": [
                        {"from": "human", "value": "<state_info>"},
                        {"from": "gpt", "value": json.dumps(valid)},
                    ],
                    "metadata": {"task": "one_leg"},
                },
                {
                    "id": "null-2d",
                    "image": ["front.png", "wrist.png"],
                    "state_info": {},
                    "conversations": [
                        {"from": "human", "value": "<state_info>"},
                        {"from": "gpt", "value": json.dumps(invalid)},
                    ],
                    "metadata": {"task": "one_leg"},
                },
            ]
        )
    )
    output_file = tmp_path / "llamafactory.json"
    args = argparse.Namespace(
        input_file=str(input_file),
        output_file=str(output_file),
        dataset_info_file=None,
        dataset_name="test",
        llamafactory_state_mode="base",
        system_prompt=None,
    )

    result = generator.convert_sharegpt_to_llamafactory(args)

    assert result["num_input_samples"] == 2
    assert result["num_samples"] == 1
    assert result["num_skipped_invalid"] == 1
    assert result["skipped"] == {
        "invalid_supervision:target_point_2d_null": 1
    }
    assert [item["id"] for item in json.loads(output_file.read_text())] == ["valid"]


def test_runtime_payload_accepts_numpy_then_validates():
    obs = {
        "skill": "pick",
        "guidance_point_2d": {"color_image2": np.array([100.0, 120.0])},
        "guidance_point": np.array([0.1, 0.2, 0.3]),
    }
    payload = generator._assistant_payload(obs, task="one_leg")
    assert generator._supervision_error(
        payload,
        front_image_shape=(240, 320, 3),
    ) is None


def test_legacy_sim_guidance_is_exported_in_robot_base():
    obs = {
        "skill": "pick",
        "guidance_point_2d": {"color_image2": np.array([100.0, 120.0])},
        "guidance_point": np.array([0.1, 0.2, 0.5]),
        "robot_state": {
            "ee_pos": np.array([0.5, 0.0, 0.1]),
            "ee_pos_sim": np.array([0.2, 0.0, 0.515]),
        },
    }

    payload = generator._assistant_payload(
        obs,
        task="one_leg",
        guidance_frame="sim-local",
    )

    np.testing.assert_allclose(payload["target_point_3d"], [0.4, 0.2, 0.085])


def test_target_rotation_6d_roundtrips_scripted_pose():
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = [0.2, 0.3, 0.4]

    rotation_6d, error, metrics = generator._target_rotation_6d_payload(
        pose,
        target_point_3d=np.array([0.2, 0.3, 0.4]),
        clean_guidance_pose=pose.copy(),
    )

    assert error is None
    np.testing.assert_allclose(
        rotation_6d,
        rotation[:2].reshape(6),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        generator._rotation_6d_to_matrix(rotation_6d),
        rotation,
        atol=1e-6,
    )
    assert metrics["roundtrip_max_error"] < 1e-6


def test_target_rotation_6d_rejects_translation_mismatch():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [0.2, 0.3, 0.4]
    _, error, _ = generator._target_rotation_6d_payload(
        pose,
        target_point_3d=[0.2, 0.3, 0.5],
    )
    assert error == "guidance_pose_translation_mismatch"


def test_enrich_existing_dataset_adds_rotation_without_media(tmp_path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [0.2, 0.3, 0.4]
    robot_state = {
        "ee_pos_sim": np.array([0.1, 0.2, 0.3]),
        "ee_quat_sim": np.array([0.0, 0.0, 0.0, 1.0]),
        "ee_pos_vel": np.zeros(3),
        "ee_ori_vel": np.zeros(3),
        "gripper_width": np.array([0.05]),
    }
    observation = {
        "color_image2": image,
        "color_image1": image,
        "robot_state": robot_state,
        "skill": "pick",
        "guidance_point_2d": {"color_image2": np.array([4.0, 5.0])},
        "guidance_point": np.array([0.2, 0.3, 0.4]),
        "guidance_pose": pose,
        "guidance_pose_clean": pose.copy(),
    }
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    pickle_path = campaign_dir / "rollout.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(
            {
                "task": "one_leg",
                "success": True,
                "action_type": "pos",
                "observations": [observation],
            },
            stream,
        )
    assistant = generator._assistant_payload(observation, task="one_leg")
    sample_id = "one_leg_00000_rollout_frame_00000"
    source_record = {
        "id": sample_id,
        "images": [
            f"images/one_leg/{sample_id}_front.png",
            f"images/one_leg/{sample_id}_wrist.png",
        ],
        "state_info": generator._state_info_payload(observation),
        "messages": [
            {"role": "system", "content": generator.TASK_SYSTEM_PROMPTS["one_leg"]},
            {"role": "user", "content": generator.DEFAULT_USER_PROMPT},
            {"role": "assistant", "content": json.dumps(assistant, sort_keys=True)},
        ],
        "metadata": {
            "task": "one_leg",
            "source_pickle": str(pickle_path),
            "rollout_index_for_task": 0,
            "frame_index": 0,
            "depth": {"front": "front.npy", "wrist": "wrist.npy"},
        },
    }
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    input_messages = input_dir / "messages.jsonl"
    input_messages.write_text(json.dumps(source_record) + "\n")
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_samples": 1,
                "samples_per_task": {"one_leg": 1},
                "schema": {"assistant_json_keys": list(assistant)},
            }
        )
    )
    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        input_messages=str(input_messages),
        source_manifest=None,
        output_dir=str(output_dir),
        source_revision="source-revision",
        annotation_source="scripted",
        expected_pickle_dir=[str(campaign_dir)],
        expected_samples=1,
        preview_samples=1,
        output_mode="error",
    )

    result = generator.enrich_existing_dataset_with_rotation_6d(args)

    assert result["num_samples"] == 1
    row = json.loads((output_dir / "messages.jsonl").read_text())
    enriched = json.loads(row["messages"][-1]["content"])
    assert enriched == {
        **assistant,
        "target_rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    assert row["metadata"]["annotation_source"] == "scripted"
    assert not (output_dir / "images").exists()
    assert not (output_dir / "depth").exists()
    llama = json.loads((output_dir / "llamafactory_base.json").read_text())
    assert len(llama) == 1
    assert "target_rotation_6d" in llama[0]["conversations"][-1]["value"]
    audit = json.loads(
        (output_dir / "rotation6d_enrichment_audit_20260831.json").read_text()
    )
    assert audit["num_source_pickles"] == 1
    assert audit["legacy_pickles_missing_top_level_annotation_source"] == 1


def test_convert_skips_null_before_writing_media(tmp_path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    robot_state = {
        "ee_pos": np.array([0.1, 0.2, 0.3]),
        "ee_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "ee_pos_vel": np.zeros(3),
        "ee_ori_vel": np.zeros(3),
        "gripper_width": np.array([0.05]),
    }
    valid = {
        "color_image2": image,
        "color_image1": image,
        "robot_state": robot_state,
        "skill": "pick",
        "guidance_point_2d": {"color_image2": np.array([4.0, 5.0])},
        "guidance_point": np.array([0.2, 0.3, 0.4]),
    }
    null_2d = dict(valid)
    null_2d["guidance_point_2d"] = {"color_image2": None}
    empty_2d = dict(valid)
    empty_2d["guidance_point_2d"] = {}
    pickle_path = tmp_path / "rollout.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(
            {
                "task": "one_leg",
                "success": True,
                "action_type": "pos",
                "observations": [valid, null_2d, empty_2d],
            },
            stream,
        )

    output_dir = tmp_path / "dataset"
    args = argparse.Namespace(
        task_rollout=["one_leg=1"],
        tasks=None,
        rollouts_per_task=None,
        input_dir=[str(pickle_path)],
        output_dir=str(output_dir),
        randomness="low",
        suffix="rgbd-only-skill",
        demo_outcome="success",
        format="all",
        llamafactory_state_mode="base",
        output_mode="error",
        frame_stride=1,
        max_frames_per_rollout=0,
        allow_legacy_eepose=False,
        save_depth_npy=False,
        system_prompt=None,
        user_prompt=generator.DEFAULT_USER_PROMPT,
    )

    manifest = generator.convert_pickles_to_vlm_sft(args)

    assert manifest["num_samples"] == 1
    assert manifest["skipped"] == {
        "invalid_supervision:target_point_2d_null": 2
    }
    assert len(list((output_dir / "images").rglob("*.png"))) == 2
    row = json.loads((output_dir / "messages.jsonl").read_text())
    assistant = json.loads(row["messages"][-1]["content"])
    assert assistant["target_point_2d"] == [4.0, 5.0]
