#!/usr/bin/env python3
"""Validate and run the depth-fixed VLM + DiT evaluation matrix.

The evaluator is deliberately never invoked directly here.  Every matrix cell
is expanded and executed through gpu-snatcher/auto_eval.sh.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable
from urllib.request import Request, urlopen

from src.eval.vlm_content_audit import audit_manifest_rollouts


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_EVAL_DEFAULT = Path("/data/hy/gpu-snatcher/auto_eval.sh")
EXPECTED_GPU_SNATCHER_COMMIT = "ebfea2d9f27bfdcea3a30791ebd6e70a05757799"
EXPECTED_VLM_REVISION = "75dc7b8a4a1dcdf6ec77398494724c7b7b3fe63e"
TASKS = ("one_leg", "round_table", "lamp")
TASK_MAX_STEPS = {task: 1000 for task in TASKS}
HISTORICAL_CLEAN_REPORT = REPO_ROOT / "reports/multi_task_condition_eval_0610.md"
HISTORICAL_CLEAN_SUCCESS = {
    # Fixed-checkpoint, low-randomness, 36-rollout clean/scripted references
    # extracted from reports/multi_task_condition_eval_0610.md.
    ("rgbd_gp", "one_leg"): (31, 36),
    ("rgbd_gp", "round_table"): (20, 36),
    ("rgbd_gp", "lamp"): (11, 36),
    ("rgbd_colored_gp", "one_leg"): (33, 36),
    ("rgbd_colored_gp", "round_table"): (10, 36),
    ("rgbd_colored_gp", "lamp"): (14, 36),
    ("rgbd_gp_skill", "one_leg"): (30, 36),
    ("rgbd_gp_skill", "round_table"): (18, 36),
    ("rgbd_gp_skill", "lamp"): (20, 36),
}
FORBIDDEN_LOG_TEXT = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "No space left on device",
    "corrupted size vs. prev_size in fastbins",
    "Killed",
)

CONDITIONS: dict[str, dict[str, Any]] = {
    "rgbd_gp": {
        "label": "rgbd+GP",
        "checkpoint": Path(
            "/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/"
            "2026-06-13/13-02-04.275134/models/"
            "icy-vortex-9_2026-06-13_13-02-27.880769/"
            "actor_chkpt_latest_3000.pt"
        ),
        "sha256": "39e6f2f44e318de3f606a81f2d2bb84c5de017ae65590b50916666edc41b340b",
        "visual_flags": ("--guidance-point-on-image", "--no-annotate-skill"),
        "expected_config": {
            "observation_type": "rgbd",
            "annotate_guidance_point": True,
            "annotate_guidance_point_colored": False,
            "annotate_skill_one_hot": False,
            "action_horizon": 8,
            "actor_name": "diffusion",
            "diffusion_model_name": "dit",
        },
    },
    "rgbd_colored_gp": {
        "label": "rgbd+colored GP",
        "checkpoint": Path(
            "/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/"
            "2026-06-18/14-59-28.908152/models/"
            "absurd-voice-2_2026-06-18_14-59-48.700671/"
            "actor_chkpt_latest_3000.pt"
        ),
        "sha256": "c82571d3246ae71f6c8591655422474a7232860c36b38996a4873ac0d283d088",
        "visual_flags": (
            "--guidance-point-on-image",
            "--guidance-point-colored",
            "--no-annotate-skill",
        ),
        "expected_config": {
            "observation_type": "rgbd",
            "annotate_guidance_point": True,
            "annotate_guidance_point_colored": True,
            "annotate_skill_one_hot": False,
            "action_horizon": 8,
            "actor_name": "diffusion",
            "diffusion_model_name": "dit",
        },
    },
    "rgbd_gp_skill": {
        "label": "rgbd+GP+skill",
        "checkpoint": Path(
            "/mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/"
            "2026-06-13/12-55-43.621615/models/"
            "fresh-tree-11_2026-06-13_12-56-10.422936/"
            "actor_chkpt_latest_3000.pt"
        ),
        "sha256": "7ec84d6d63ccc2307e0efc115bfc9ddcedbb13e2fdd97c2be10bfc5e9e741887",
        "visual_flags": (
            "--guidance-point-on-image",
            "--annotate-skill",
            "--skill-on-image",
        ),
        "expected_config": {
            "observation_type": "rgbd",
            "annotate_guidance_point": True,
            "annotate_guidance_point_colored": False,
            "annotate_skill_one_hot": True,
            "action_horizon": 8,
            "actor_name": "diffusion",
            "diffusion_model_name": "dit",
        },
    },
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_token(path: Path | None) -> str | None:
    token = os.environ.get("VLM_API_TOKEN")
    if token:
        return token
    if path is None or not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("VLM_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def _check_ready(base_url: str, token: str | None, timeout: float) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(f"{base_url.rstrip('/')}/health/ready", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    actual = (
        payload.get("status"),
        payload.get("policy_version"),
        payload.get("model_revision"),
    )
    expected = ("ready", 3, EXPECTED_VLM_REVISION)
    if actual != expected:
        raise RuntimeError(
            "VLM readiness contract mismatch: "
            f"status/policy/revision={actual!r}, expected={expected!r}"
        )
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_CHECKPOINT_AUDIT_CODE = r"""
import json
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
cfg = checkpoint["config"]
data = cfg.get("data") or {}
actor = cfg.get("actor") or {}
diffusion = actor.get("diffusion_model") or {}
result = {
    "observation_type": cfg.get("observation_type"),
    "annotate_guidance_point": bool(data.get("annotate_guidance_point", False)),
    "annotate_guidance_point_colored": bool(data.get("annotate_guidance_point_colored", False)),
    "annotate_skill_one_hot": bool(data.get("annotate_skill_one_hot", False)),
    "skill_dim": cfg.get("skill_dim"),
    "action_horizon": cfg.get("action_horizon", actor.get("action_horizon")),
    "actor_name": cfg.get("actor_name", actor.get("name")),
    "diffusion_model_name": diffusion.get("name"),
}
print(json.dumps(result, sort_keys=True))
"""


def audit_checkpoints(rr_python: Path, conditions: Iterable[str]) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for condition in conditions:
        spec = CONDITIONS[condition]
        checkpoint = spec["checkpoint"]
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint for {condition}: {checkpoint}")
        actual_sha = _sha256(checkpoint)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch for {condition}: {actual_sha} "
                f"!= {spec['sha256']}"
            )
        completed = subprocess.run(
            [str(rr_python), "-c", _CHECKPOINT_AUDIT_CODE, str(checkpoint)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Checkpoint config audit failed for {condition}: {completed.stderr.strip()}"
            )
        actual_config = json.loads(completed.stdout.strip().splitlines()[-1])
        conflicts = {
            key: {"actual": actual_config.get(key), "expected": expected}
            for key, expected in spec["expected_config"].items()
            if actual_config.get(key) != expected
        }
        if conflicts:
            raise RuntimeError(
                f"Checkpoint config conflict for {condition}: "
                f"{json.dumps(conflicts, ensure_ascii=False, sort_keys=True)}"
            )
        audits[condition] = {
            "checkpoint": str(checkpoint),
            "sha256": actual_sha,
            "config": actual_config,
        }
    return audits


def _option(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def build_auto_eval_command(
    *,
    args: argparse.Namespace,
    condition: str,
    task: str,
    summary_path: Path,
    rollout_suffix: str,
    print_command: bool,
    annotation_source: str = "vlm",
) -> list[str]:
    spec = CONDITIONS[condition]
    n_envs = int(getattr(args, "n_envs", 3))
    preview_overlay = getattr(args, "stage", None) == "preview"
    visual_flags = (
        ("--guidance-point-on-image", "--annotate-skill", "--skill-on-image")
        if preview_overlay
        else spec["visual_flags"]
    )
    command = [
        str(args.auto_eval),
        "--steps",
        "eval",
        "--local-path",
        str(REPO_ROOT),
        "--overwrite-wt-path",
        str(spec["checkpoint"]),
        "--task",
        task,
        "--n-envs",
        str(n_envs),
        "--n-rollouts",
        str(args.n_rollouts),
        "--randomness",
        "low",
        "--max-rollout-steps",
        "1000",
        "--annotation-source",
        annotation_source,
        "--tracking-metric-type",
        "pose",
        "--task-summary-out",
        str(summary_path),
        "--rollout-suffix-model-name",
        rollout_suffix,
        "--gpu-id",
        str(args.gpu),
        *visual_flags,
    ]
    if annotation_source == "vlm":
        command.extend(
            [
                "--vlm-base-url",
                args.vlm_base_url,
                "--vlm-timeout-seconds",
                "30",
                "--vlm-query-interval",
                "0",
                "--vlm-noise-projection-samples",
                "200",
            ]
        )
    if print_command:
        command.append("--print-command")
    return command


def _extract_expanded_command(output: str) -> list[str]:
    candidates = [line.strip() for line in output.splitlines() if line.startswith("python ")]
    if not candidates:
        raise RuntimeError("auto_eval.sh did not print an expanded evaluate_model command")
    command = shlex.split(candidates[-1])
    if command[:4] != ["python", "-m", "src.eval.evaluate_model", "--n-envs"]:
        raise RuntimeError(f"Unexpected expanded command prefix: {command[:5]!r}")
    return command


def _extract_logged_command(output: str) -> list[str] | None:
    marker = "Evaluation command: "
    matches = [line.split(marker, 1)[1] for line in output.splitlines() if marker in line]
    return shlex.split(matches[-1]) if matches else None


def validate_expanded_command(
    command: list[str],
    *,
    condition: str,
    task: str,
    n_rollouts: int,
    summary_path: Path,
    rollout_suffix: str,
    vlm_base_url: str,
    n_envs: int = 3,
    preview_overlay: bool = False,
) -> None:
    expected_options = {
        "--n-envs": str(n_envs),
        "--n-rollouts": str(n_rollouts),
        "-f": task,
        "--max-rollout-steps": "1000",
        "--action-type": "pos",
        "--observation-space": "image",
        "--randomness": "low",
        "--annotation-source": "vlm",
        "--tracking-metric-type": "pose",
        "--vlm-base-url": vlm_base_url,
        "--vlm-timeout-seconds": "30",
        "--vlm-query-interval": "0",
        "--vlm-noise-projection-samples": "200",
        "--task-summary-out": str(summary_path),
        "--rollout-suffix-model-name": rollout_suffix,
        "--wt-path": str(CONDITIONS[condition]["checkpoint"]),
        "--if-exists": "append",
    }
    errors = [
        f"{name}={_option(command, name)!r}, expected {expected!r}"
        for name, expected in expected_options.items()
        if _option(command, name) != expected
    ]
    for required in ("--save-rollouts", "--save-failures", "--save-depth-image"):
        if required not in command:
            errors.append(f"missing {required}")
    for forbidden in (
        "--compress-pickles",
        "--output-only-video",
        "--annotation-source scripted",
    ):
        if forbidden in command or forbidden in " ".join(command):
            errors.append(f"forbidden token {forbidden}")
    visual_flags = (
        ("--guidance-point-on-image", "--annotate-skill", "--skill-on-image")
        if preview_overlay
        else CONDITIONS[condition]["visual_flags"]
    )
    for flag in visual_flags:
        if flag != "--no-annotate-skill" and flag not in command:
            errors.append(f"missing condition visualization flag {flag}")
    if (
        not preview_overlay
        and condition != "rgbd_gp_skill"
        and "--annotate-skill" in command
    ):
        errors.append("unexpected --annotate-skill")
    if errors:
        raise RuntimeError(
            f"Expanded command validation failed for {condition}/{task}: "
            + "; ".join(errors)
        )


def _normalized_smoke_formal_command(command: list[str]) -> list[str]:
    normalized = list(command)
    for option, replacement in (
        ("--n-rollouts", "<N_ROLLOUTS>"),
        ("--task-summary-out", "<SUMMARY_PATH>"),
        ("--rollout-suffix-model-name", "<ROLLOUT_SUFFIX>"),
    ):
        try:
            normalized[normalized.index(option) + 1] = replacement
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"Command is missing comparison option {option}") from exc
    return normalized


def _require_formal_smoke_gate(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("Formal matrix requires --smoke-manifest from a completed smoke")
    smoke = json.loads(path.read_text())
    gate = smoke.get("smoke_gate") or {}
    if gate.get("status") != "passed":
        raise RuntimeError(f"Smoke automatic gate did not pass: {gate.get('failures')}")
    manual = gate.get("manual_review") or {}
    if manual.get("status") != "passed":
        raise RuntimeError(
            "Formal matrix requires smoke_gate.manual_review.status=passed after "
            "pickle/MP4 and historical-reference review"
        )
    return smoke


def _formal_gate_record(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.stage != "formal":
        return None
    if args.allow_formal_without_smoke:
        return {
            "mode": "explicit_user_approved_bypass",
            "smoke_manifest": None,
            "approval_note": args.formal_approval_note,
        }
    return {
        "mode": "smoke_manifest",
        "smoke_manifest": str(args.smoke_manifest.resolve()),
        "approval_note": None,
    }


def _run_capture(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
    return process.wait(), "".join(captured)


def _summary_error(path: Path, n_rollouts: int) -> str | None:
    if not path.is_file():
        return "summary missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"summary unreadable: {exc}"
    if int(payload.get("n_rollouts", -1)) != n_rollouts:
        return f"n_rollouts={payload.get('n_rollouts')!r}"
    if not isinstance(payload.get("n_success"), int):
        return "n_success missing"
    tracking = payload.get("tracking_error") or {}
    if not (tracking.get("overall") or {}).get("count"):
        return "tracking metrics empty"
    point = payload.get("vlm_point_error") or {}
    if not (((point.get("all") or {}).get("overall") or {}).get("count_valid")):
        return "VLM point metrics empty"
    if payload.get("vlm_model_revision") != EXPECTED_VLM_REVISION:
        return f"summary VLM revision mismatch: {payload.get('vlm_model_revision')!r}"
    return None


def _manifest_rows(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        for task in args.tasks:
            key = f"{condition}__{task}"
            rollout_suffix = f"{args.namespace}/{condition}/{task}"
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITIONS[condition]["label"],
                    "task": task,
                    "checkpoint": str(CONDITIONS[condition]["checkpoint"]),
                    "summary_path": str(output_dir / "summaries" / f"{key}.json"),
                    "stdout_path": str(output_dir / "stdout" / f"{key}.log"),
                    "print_stdout_path": str(output_dir / "expanded" / f"{key}.log"),
                    "rollout_suffix": rollout_suffix,
                    "status": "pending",
                }
            )
    return rows


def _git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _child_env(args: argparse.Namespace, token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    rr_library_dir = args.rr_python.parent.parent / "lib"
    existing_library_path = env.get("LD_LIBRARY_PATH", "")
    if rr_library_dir.is_dir():
        env["LD_LIBRARY_PATH"] = (
            f"{existing_library_path}:{rr_library_dir}"
            if existing_library_path
            else str(rr_library_dir)
        )
    nvidia_user_lib_dir = env.get("NVIDIA_USER_LIB_DIR", "").strip()
    if nvidia_user_lib_dir:
        library_dir = Path(nvidia_user_lib_dir)
        required = (library_dir / "libcuda.so.1", library_dir / "libnvidia-ml.so.1")
        if not all(path.exists() for path in required):
            raise RuntimeError(
                "NVIDIA_USER_LIB_DIR does not contain matching libcuda/NVML SONAMEs: "
                f"{library_dir}"
            )
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{library_dir}:{existing}" if existing else str(library_dir)
        )
        matching_driver_preload = ":".join(str(path) for path in required)
        existing_preload = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = (
            f"{matching_driver_preload}:{existing_preload}"
            if existing_preload
            else matching_driver_preload
        )
    env["PYTHONUNBUFFERED"] = "1"
    env["DATA_DIR_RAW"] = str(args.data_dir_raw.resolve())
    env["VLM_GUIDANCE_URL"] = args.vlm_base_url
    if token:
        env["VLM_API_TOKEN"] = token
    return env


def print_phase(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {manifest_path}")
    gpu_commit = _git_commit(args.auto_eval.parent)
    if gpu_commit != EXPECTED_GPU_SNATCHER_COMMIT:
        raise SystemExit(
            f"gpu-snatcher HEAD {gpu_commit} != required {EXPECTED_GPU_SNATCHER_COMMIT}"
        )
    audits = audit_checkpoints(args.rr_python, args.conditions)
    smoke_manifest = None
    if args.stage == "formal" and not args.allow_formal_without_smoke:
        smoke_manifest = _require_formal_smoke_gate(args.smoke_manifest)
    token = _read_token(args.token_file)
    readiness = _check_ready(args.vlm_base_url, token, 30.0)
    env = _child_env(args, token)
    rows = _manifest_rows(args, output_dir)
    summary_paths = {row["summary_path"] for row in rows}
    suffixes = {row["rollout_suffix"] for row in rows}
    if len(summary_paths) != len(rows) or len(suffixes) != len(rows):
        raise SystemExit("Summary paths and rollout suffixes must be unique")

    for row in rows:
        command = build_auto_eval_command(
            args=args,
            condition=row["condition"],
            task=row["task"],
            summary_path=Path(row["summary_path"]),
            rollout_suffix=row["rollout_suffix"],
            print_command=True,
        )
        completed = _run_capture(command, env=env)
        print_log = Path(row["print_stdout_path"])
        print_log.parent.mkdir(parents=True, exist_ok=True)
        print_log.write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(
                f"auto_eval command expansion failed for {row['condition']}/{row['task']}:\n"
                f"{completed.stdout}"
            )
        expanded = _extract_expanded_command(completed.stdout)
        validate_expanded_command(
            expanded,
            condition=row["condition"],
            task=row["task"],
            n_rollouts=args.n_rollouts,
            summary_path=Path(row["summary_path"]),
            rollout_suffix=row["rollout_suffix"],
            vlm_base_url=args.vlm_base_url,
            n_envs=args.n_envs,
            preview_overlay=args.stage == "preview",
        )
        row["auto_eval_command"] = command
        row["expanded_evaluate_command"] = expanded
        row["command_validation"] = "passed"
        if smoke_manifest is not None:
            smoke_row = next(
                candidate
                for candidate in smoke_manifest["runs"]
                if candidate["condition"] == row["condition"]
                and candidate["task"] == row["task"]
            )
            if _normalized_smoke_formal_command(expanded) != _normalized_smoke_formal_command(
                smoke_row["expanded_evaluate_command"]
            ):
                raise RuntimeError(
                    "Formal/smoke expanded argv differ beyond n_rollouts, summary path, "
                    f"and rollout suffix for {row['condition']}/{row['task']}"
                )
            row["smoke_argv_comparison"] = "passed"
        print(f"PASS {row['condition']}/{row['task']}: {shlex.join(command)}")
        print(f"  expanded: {shlex.join(expanded)}")

    manifest = {
        "version": 3,
        "stage": args.stage,
        "namespace": args.namespace,
        "created_at": _timestamp(),
        "gpu_snatcher_commit": gpu_commit,
        "robust_rearrangement_commit": _git_commit(REPO_ROOT),
        "auto_eval_path": str(args.auto_eval),
        "vlm_base_url": args.vlm_base_url,
        "vlm_readiness": readiness,
        "expected_vlm_revision": EXPECTED_VLM_REVISION,
        "checkpoint_audits": audits,
        "n_envs": args.n_envs,
        "n_rollouts_per_task": args.n_rollouts,
        "total_requested_rollouts": len(rows) * args.n_rollouts,
        "randomness": "low",
        "max_rollout_steps": 1000,
        "tracking_metric_type": "pose",
        "vlm_noise_projection_samples": 200,
        "data_dir_raw": str(args.data_dir_raw.resolve()),
        "commands_validated": True,
        "smoke_manifest": str(args.smoke_manifest.resolve()) if args.smoke_manifest else None,
        "formal_gate": _formal_gate_record(args),
        "runs": rows,
    }
    _atomic_json(manifest_path, manifest)
    print(f"Validated command manifest: {manifest_path}")
    return 0


def _load_execution_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    manifest_path = args.output_dir.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Run --phase print first; manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("commands_validated"):
        raise SystemExit("Manifest does not contain validated expanded commands")
    expected = {
        "stage": args.stage,
        "namespace": args.namespace,
        "n_rollouts_per_task": args.n_rollouts,
        "n_envs": args.n_envs,
        "randomness": "low",
        "max_rollout_steps": 1000,
        "vlm_noise_projection_samples": 200,
        "data_dir_raw": str(args.data_dir_raw.resolve()),
    }
    if args.stage == "formal":
        expected["smoke_manifest"] = (
            str(args.smoke_manifest.resolve()) if args.smoke_manifest else None
        )
        expected["formal_gate"] = _formal_gate_record(args)
    conflicts = {
        key: {"manifest": manifest.get(key), "runtime": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if conflicts:
        raise SystemExit(f"Runtime/manifest mismatch: {json.dumps(conflicts)}")
    if any(row.get("status") != "pending" for row in manifest.get("runs", [])):
        raise SystemExit("Refusing to resume or append a previously started matrix")
    return manifest_path, manifest


def _smoke_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = manifest.get("runs", [])
    for row in rows:
        if row.get("status") != "complete" or row.get("return_code") != 0:
            failures.append(f"{row['condition']}/{row['task']} did not complete cleanly")
            continue
        payload = json.loads(Path(row["summary_path"]).read_text())
        smoke_success = int(payload.get("n_success", 0))
        historical_success, historical_total = HISTORICAL_CLEAN_SUCCESS[
            (row["condition"], row["task"])
        ]
        if row["condition"] == "rgbd_gp" and row["task"] == "one_leg":
            if smoke_success < 2:
                failures.append("rgbd_gp/one_leg success is below required 2/3")
        elif historical_success / historical_total > 0.5 and smoke_success == 0:
            failures.append(
                f"{row['condition']}/{row['task']} is 0/3 despite historical clean "
                f"success {historical_success}/{historical_total}"
            )
    condition_task_success = {
        f"{row['condition']}/{row['task']}": json.loads(Path(row["summary_path"]).read_text()).get("n_success")
        for row in rows
        if row.get("status") == "complete"
    }
    content_audit = manifest.get("vlm_content_audit") or {}
    if content_audit.get("status") != "passed":
        failed_rows = content_audit.get("failed_rows") or ["audit missing"]
        failures.append(
            "fresh VLM guidance content audit failed: " + ", ".join(failed_rows)
        )
    return {
        "status": "passed" if not failures else "failed",
        "checked_at": _timestamp(),
        "failures": failures,
        "condition_task_success": condition_task_success,
        "historical_clean_reference": {
            "source": str(HISTORICAL_CLEAN_REPORT),
            "cells": {
                f"{condition}/{task}": {
                    "success": success,
                    "total": total,
                    "rate": success / total,
                }
                for (condition, task), (success, total) in HISTORICAL_CLEAN_SUCCESS.items()
            },
        },
        "automatic_checks": {
            "all_return_codes_zero": not any(row.get("return_code") for row in rows),
            "all_summaries_complete": not any(row.get("summary_error") for row in rows),
            "depth_contract_logged": all(row.get("depth_contract_logged") for row in rows),
            "vlm_revision_constant": all(
                row.get("vlm_readiness", {}).get("model_revision") == EXPECTED_VLM_REVISION
                for row in rows
            ),
            "vlm_fresh_point_dynamics": content_audit.get("status") == "passed",
        },
        "vlm_content_audit": content_audit,
        "manual_visual_review_required": True,
        "manual_gate_note": (
            "Marker/coordinate, colored-GP color, skill text, historical-reference, "
            "and cross-condition-collapse checks require pickle/MP4 inspection before formal launch."
        ),
    }


def run_phase(args: argparse.Namespace) -> int:
    manifest_path, manifest = _load_execution_manifest(args)
    token = _read_token(args.token_file)
    env = _child_env(args, token)
    audit_checkpoints(args.rr_python, args.conditions)
    if args.stage == "formal" and not args.allow_formal_without_smoke:
        _require_formal_smoke_gate(args.smoke_manifest)
    manifest["nvidia_user_lib_dir"] = env.get("NVIDIA_USER_LIB_DIR")
    gpu_preflight = subprocess.run(
        [
            str(args.rr_python),
            "-c",
            (
                "import isaacgym; import torch; "
                "assert torch.cuda.is_available(); "
                "x=torch.ones(1, device='cuda:0'); "
                "print(torch.cuda.get_device_name(0))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    manifest["gpu_preflight"] = {
        "checked_at": _timestamp(),
        "return_code": gpu_preflight.returncode,
        "device": gpu_preflight.stdout.strip(),
        "stderr": gpu_preflight.stderr.strip(),
    }
    _atomic_json(manifest_path, manifest)
    if gpu_preflight.returncode:
        raise RuntimeError(
            "CUDA preflight failed; no rollout was started: "
            f"{gpu_preflight.stderr.strip()}"
        )

    for row in manifest["runs"]:
        readiness = _check_ready(args.vlm_base_url, token, 30.0)
        row["vlm_readiness"] = readiness
        row["started_at"] = _timestamp()
        row["status"] = "running"
        command = [arg for arg in row["auto_eval_command"] if arg != "--print-command"]
        row["executed_auto_eval_command"] = command
        _atomic_json(manifest_path, manifest)
        print(f"\n=== {row['condition_label']} / {row['task']} ===", flush=True)
        return_code, output = _run_and_tee(command, Path(row["stdout_path"]), env)
        row["return_code"] = return_code
        row["finished_at"] = _timestamp()
        try:
            readiness_after = _check_ready(args.vlm_base_url, token, 30.0)
            row["vlm_readiness_after"] = readiness_after
            revision_changed = (
                readiness_after.get("model_revision")
                != readiness.get("model_revision")
            )
        except Exception as exc:
            row["vlm_readiness_after"] = {"error": str(exc)}
            revision_changed = True
        logged_command = _extract_logged_command(output)
        row["logged_evaluate_command"] = logged_command
        row["depth_contract_logged"] = (
            "RGBD observation contract: depth_image1=" in output
            and "depth_image2=" in output
        )
        row["forbidden_log_matches"] = [text for text in FORBIDDEN_LOG_TEXT if text in output]
        summary_error = _summary_error(Path(row["summary_path"]), args.n_rollouts)
        row["summary_error"] = summary_error
        command_mismatch = logged_command != row["expanded_evaluate_command"]
        row["command_mismatch"] = command_mismatch
        failure_reasons = []
        if return_code:
            failure_reasons.append(f"return_code={return_code}")
        if summary_error:
            failure_reasons.append(summary_error)
        if command_mismatch:
            failure_reasons.append("runtime evaluator command differs from validated command")
        if row["forbidden_log_matches"]:
            failure_reasons.append(f"forbidden log text: {row['forbidden_log_matches']}")
        if revision_changed:
            failure_reasons.append("VLM readiness/revision changed during row")
        if not row["depth_contract_logged"]:
            failure_reasons.append("RGBD depth observation contract log missing")
        row["failure_reasons"] = failure_reasons
        row["status"] = "failed" if failure_reasons else "complete"
        _atomic_json(manifest_path, manifest)
        if failure_reasons:
            print(
                f"STOP {row['condition']}/{row['task']}: {'; '.join(failure_reasons)}",
                file=sys.stderr,
            )
            return 1

    if args.stage == "smoke":
        content_audit_path = args.output_dir.resolve() / "vlm_content_audit.json"
        content_audit = audit_manifest_rollouts(manifest)
        _atomic_json(content_audit_path, content_audit)
        manifest["vlm_content_audit"] = {
            "status": content_audit["status"],
            "path": str(content_audit_path),
            "failed_rows": content_audit["failed_rows"],
        }
        gate = _smoke_gate(manifest)
        manifest["smoke_gate"] = gate
        _atomic_json(args.output_dir.resolve() / "smoke_gate.json", gate)
        _atomic_json(manifest_path, manifest)
        if gate["status"] != "passed":
            return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("print", "run"), required=True)
    parser.add_argument(
        "--stage", choices=("preview", "smoke", "formal"), required=True
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir-raw", type=Path, required=True)
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        help="Required for formal: completed smoke manifest with automatic and manual gates passed.",
    )
    parser.add_argument(
        "--allow-formal-without-smoke",
        action="store_true",
        help=(
            "Explicitly bypass the formal smoke-manifest gate after external user review. "
            "Requires --formal-approval-note and is recorded in the manifest."
        ),
    )
    parser.add_argument(
        "--formal-approval-note",
        help="Approval provenance recorded when --allow-formal-without-smoke is used.",
    )
    parser.add_argument("--auto-eval", type=Path, default=AUTO_EVAL_DEFAULT)
    parser.add_argument("--rr-python", type=Path, default=Path("/home/hy/anaconda3/envs/rr/bin/python"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--vlm-base-url",
        default=os.environ.get("VLM_GUIDANCE_URL", "http://10.71.106.240:8000"),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("/mnt/nas/share/home/hy/vlm-guidance/server.env"),
    )
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    if args.stage == "preview":
        args.n_rollouts = 1
        args.n_envs = 1
        if args.conditions != ["rgbd_gp"]:
            raise SystemExit("Preview requires --conditions rgbd_gp")
    else:
        args.n_rollouts = 3 if args.stage == "smoke" else 36
        args.n_envs = 3
        if args.conditions != list(CONDITIONS) or args.tasks != list(TASKS):
            raise SystemExit("Smoke/formal require the complete fixed 3x3 matrix")
    if args.allow_formal_without_smoke:
        if args.stage != "formal":
            raise SystemExit("--allow-formal-without-smoke is only valid for formal")
        if args.smoke_manifest is not None:
            raise SystemExit(
                "--allow-formal-without-smoke and --smoke-manifest are mutually exclusive"
            )
        if not (args.formal_approval_note or "").strip():
            raise SystemExit(
                "--allow-formal-without-smoke requires --formal-approval-note"
            )
    elif args.formal_approval_note is not None:
        raise SystemExit(
            "--formal-approval-note requires --allow-formal-without-smoke"
        )
    if args.gpu < 0:
        raise SystemExit("--gpu must be non-negative")
    if not args.auto_eval.is_file():
        raise SystemExit(f"auto_eval.sh not found: {args.auto_eval}")
    if not args.rr_python.is_file():
        raise SystemExit(f"rr Python not found: {args.rr_python}")
    return args


def main() -> int:
    args = parse_args()
    return print_phase(args) if args.phase == "print" else run_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
