import pickle

import numpy as np

from src.eval.vlm_content_audit import audit_manifest_rollouts


def _write_rollout(root, *, collapsed):
    path = (
        root
        / "raw/diffik/sim/one_leg/rollout/low/rgbd-point-vlm"
        / "smoke/rgbd_gp/one_leg/success/episode.pkl"
    )
    path.parent.mkdir(parents=True)
    observations = []
    latest_query = 0
    for frame_idx in range(160):
        if frame_idx % 8 == 0:
            latest_query = frame_idx
        phase = (latest_query // 8) % 2
        oracle = np.asarray((150.0, 153.0) if phase == 0 else (240.0, 140.0))
        vlm = np.asarray((224.0, 143.0)) if collapsed else oracle.copy()
        observations.append(
            {
                "color_image2": np.full((2, 2, 3), frame_idx % 255, dtype=np.uint8),
                "skill": "pick" if phase == 0 else "place",
                "oracle_skill": "pick" if phase == 0 else "place",
                "guidance_point_2d": {"color_image2": vlm},
                "oracle_guidance_point_2d": {"color_image2": oracle},
                "vlm_annotation": {
                    "cache_age_steps": frame_idx - latest_query,
                    "query_step": latest_query,
                    "request_id": f"env0-step{latest_query}",
                    "model_revision": "revision",
                    "point_1000": vlm / np.asarray((319 / 1000, 239 / 1000)),
                },
            }
        )
    with path.open("wb") as stream:
        pickle.dump({"observations": observations}, stream)


def _manifest(root):
    return {
        "data_dir_raw": str(root),
        "randomness": "low",
        "n_rollouts_per_task": 1,
        "runs": [
            {
                "condition": "rgbd_gp",
                "task": "one_leg",
                "rollout_suffix": "smoke/rgbd_gp/one_leg",
            }
        ],
    }


def test_content_audit_rejects_fresh_point_collapse(tmp_path):
    _write_rollout(tmp_path, collapsed=True)
    audit = audit_manifest_rollouts(_manifest(tmp_path))
    assert audit["status"] == "failed"
    row = audit["rows"][0]
    assert row["fresh_query_count"] == 20
    assert row["vlm_to_gt_spread_ratio"] == 0.0
    assert row["transitions"]["large_gt_transition_count"] == 19
    assert any("spread is too small" in reason for reason in row["failures"])


def test_content_audit_accepts_responsive_fresh_points(tmp_path):
    _write_rollout(tmp_path, collapsed=False)
    audit = audit_manifest_rollouts(_manifest(tmp_path))
    assert audit["status"] == "passed"
    row = audit["rows"][0]
    assert row["vlm_to_gt_spread_ratio"] == 1.0
    assert row["transitions"]["vlm_to_gt_response_ratio"] == 1.0
