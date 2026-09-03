#!/usr/bin/env python3
"""Run the fixed local ckpts_ver2 VLM + grasp-part DiT evaluation matrix."""

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
import time
from typing import Any
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("one_leg", "round_table", "lamp")
EXPECTED_VLM_REVISION = "37ed01843096ee22e65002b416ed0d20ce0f3e34"
VLM_POLICY_VERSION = 4
VLM_OUTPUT_SCHEMA = "skill_point_rotation6d"
MAX_SAVED_ROLLOUTS = 10
CONDITIONS: dict[str, dict[str, Any]] = {
    "grasp_part": {
        "label": "rgbd+grasp-part",
        "checkpoint": REPO_ROOT
        / "checkpoints/bc/one_leg+round_table+lamp/low/"
        "multi-task-rgbd-skill-low-grasp-annotation_morning-glitter-1_last_.pt",
        "sha256": "f67f5abe46bfe2eb65049fa6f46eed4acb1313b4d3b8e4669edfd7063adfa801",
        "flags": ("--grasp-part-annotate", "--annotate-skill"),
        "colored": False,
    },
    "grasp_part_colored": {
        "label": "rgbd+grasp-part-colored",
        "checkpoint": REPO_ROOT
        / "checkpoints/bc/one_leg+round_table+lamp/low/"
        "multi-task-rgbd-skill-low-grasp-annotation_eternal-cosmos-2_last_.pt",
        "sha256": "65d57a320c5cab81a335ac0b2c45d14ed1c4876462566da78538cd9998b8bcfe",
        "flags": (
            "--grasp-part-annotate",
            "--annotate-skill",
            "--guidance-point-colored",
            "--grasp-annotation-colored",
        ),
        "colored": True,
    },
}
FORBIDDEN_LOG_TEXT = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "No space left on device",
    "Killed",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _git_provenance(repo: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"path": str(repo), "head": head, "dirty": bool(status), "status": status}


def _check_ready(base_url: str, token: str | None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(f"{base_url.rstrip('/')}/health/ready", headers=headers)
    with urlopen(request, timeout=30.0) as response:
        payload = json.load(response)
    expected = {
        "status": "ready",
        "policy_version": VLM_POLICY_VERSION,
        "output_schema": VLM_OUTPUT_SCHEMA,
        "model_revision": EXPECTED_VLM_REVISION,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"VLM readiness mismatch: {mismatches}")
    return payload


def _checkpoint_audits(conditions: list[str], rr_python: Path) -> dict[str, Any]:
    code = (
        "import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu')['config']; "
        "d=c.get('data') or {}; a=c.get('actor') or {}; "
        "print(json.dumps({'observation_type':c.get('observation_type'),"
        "'grasp_part':bool(d.get('annotate_grasp_part',d.get('grasp_part_annotate',False))),"
        "'guidance_colored':bool(d.get('annotate_guidance_point_colored',False)),"
        "'grasp_colored':bool(d.get('annotate_grasp_colored',d.get('annotate_grasp_annotation_colored',False))),"
        "'skill_one_hot':bool(d.get('annotate_skill_one_hot',False)),"
        "'action_horizon':c.get('action_horizon',a.get('action_horizon'))},sort_keys=True))"
    )
    output = {}
    for condition in conditions:
        spec = CONDITIONS[condition]
        checkpoint = spec["checkpoint"]
        actual_sha = _sha256(checkpoint)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint}")
        completed = subprocess.run(
            [str(rr_python), "-c", code, str(checkpoint)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        config = json.loads(completed.stdout.splitlines()[-1])
        expected = {
            "observation_type": "rgbd",
            "grasp_part": True,
            "guidance_colored": spec["colored"],
            "grasp_colored": spec["colored"],
            "skill_one_hot": False,
            "action_horizon": 8,
        }
        if config != expected:
            raise RuntimeError(f"checkpoint config mismatch for {condition}: {config} != {expected}")
        output[condition] = {
            "path": str(checkpoint),
            "sha256": actual_sha,
            "config": config,
        }
    return output


def _auto_eval_command(args, condition: str, task: str, summary: Path, suffix: str):
    return [
        str(args.auto_eval),
        "--steps",
        "eval",
        "--local-path",
        str(REPO_ROOT),
        "--overwrite-wt-path",
        str(CONDITIONS[condition]["checkpoint"]),
        "--task",
        task,
        "--n-envs",
        str(args.n_envs),
        "--n-rollouts",
        str(args.n_rollouts),
        "--randomness",
        "low",
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--max-saved-rollouts",
        str(MAX_SAVED_ROLLOUTS),
        "--annotation-source",
        "vlm",
        "--tracking-metric-type",
        "pose",
        "--task-summary-out",
        str(summary),
        "--rollout-suffix-model-name",
        suffix,
        "--gpu-id",
        str(args.gpu),
        "--vlm-base-url",
        args.vlm_base_url,
        "--vlm-timeout-seconds",
        str(args.vlm_timeout_seconds),
        "--vlm-query-interval",
        "0",
        "--vlm-noise-projection-samples",
        "200",
        *CONDITIONS[condition]["flags"],
    ]


def _expanded_command(output: str) -> list[str]:
    lines = [line for line in output.splitlines() if line.startswith("python ")]
    if not lines:
        raise RuntimeError("auto_eval.sh did not print an evaluator command")
    return shlex.split(lines[-1])


def _validate_expanded(command: list[str], args, condition: str, task: str) -> None:
    joined = " ".join(command)
    required = (
        "--save-depth-image",
        "--save-rollouts",
        "--save-failures",
        "--grasp-part-annotate",
        "--annotate-skill",
        "--annotation-source",
        "--tracking-metric-type",
    )
    missing = [item for item in required if item not in command]
    if condition == "grasp_part_colored":
        missing.extend(
            item
            for item in ("--guidance-point-colored", "--grasp-annotation-colored")
            if item not in command
        )
    elif "--guidance-point-colored" in command or "--grasp-annotation-colored" in command:
        missing.append("unexpected colored flag")
    expected_pairs = {
        "--n-envs": str(args.n_envs),
        "--n-rollouts": str(args.n_rollouts),
        "--max-rollout-steps": str(args.max_rollout_steps),
        "-f": task,
        "--wt-path": str(CONDITIONS[condition]["checkpoint"]),
    }
    for option, value in expected_pairs.items():
        if option not in command or command[command.index(option) + 1] != value:
            missing.append(f"{option}={value}")
    if "--compress-pickles" in joined or missing:
        raise RuntimeError(f"expanded command validation failed: {missing}")


def _child_env(args) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "DATA_DIR_RAW": str(args.data_dir_raw.resolve()),
            "VLM_GUIDANCE_URL": args.vlm_base_url,
            "VLM_POLICY_VERSION": str(VLM_POLICY_VERSION),
            "VLM_OUTPUT_SCHEMA": VLM_OUTPUT_SCHEMA,
        }
    )
    rr_lib = args.rr_python.parent.parent / "lib"
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{rr_lib}:{current}" if current else str(rr_lib)
    return env


def _gpu_memory_free_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def _wait_for_gpu_quiescence(
    *, baseline_free_mib: int, tolerance_mib: int = 256, timeout_seconds: float = 60.0
) -> dict[str, Any]:
    """Wait for a completed evaluator's CUDA context before the next cell."""

    target = max(512, int(baseline_free_mib) - int(tolerance_mib))
    started = time.monotonic()
    consecutive = 0
    samples = []
    while True:
        free_mib = _gpu_memory_free_mib()
        samples.append(free_mib)
        consecutive = consecutive + 1 if free_mib >= target else 0
        if consecutive >= 3:
            return {
                "baseline_free_mib": int(baseline_free_mib),
                "target_free_mib": target,
                "final_free_mib": free_mib,
                "wait_seconds": time.monotonic() - started,
                "sample_count": len(samples),
                "minimum_free_mib": min(samples),
            }
        if time.monotonic() - started >= timeout_seconds:
            raise RuntimeError(
                "GPU memory did not return to the pre-cell baseline: "
                f"target={target} MiB, samples={samples[-10:]}"
            )
        time.sleep(0.5)


def _print_phase(args) -> int:
    manifest_path = args.output_dir.resolve() / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite {manifest_path}")
    env = _child_env(args)
    readiness = _check_ready(args.vlm_base_url, os.environ.get("VLM_API_TOKEN"))
    audits = _checkpoint_audits(args.conditions, args.rr_python)
    rows = []
    for condition in args.conditions:
        for task in args.tasks:
            key = f"{condition}__{task}"
            summary = args.output_dir.resolve() / "summaries" / f"{key}.json"
            suffix = f"{args.namespace}/{condition}/{task}"
            command = _auto_eval_command(args, condition, task, summary, suffix)
            preview = subprocess.run(
                [*command, "--print-command"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if preview.returncode:
                raise RuntimeError(preview.stdout)
            expanded = _expanded_command(preview.stdout)
            _validate_expanded(expanded, args, condition, task)
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITIONS[condition]["label"],
                    "task": task,
                    "summary_path": str(summary),
                    "stdout_path": str(args.output_dir.resolve() / "stdout" / f"{key}.log"),
                    "rollout_suffix": suffix,
                    "auto_eval_command": command,
                    "expanded_evaluate_command": expanded,
                    "status": "pending",
                }
            )
    hy_root = REPO_ROOT / "logs/vlm_grasp_ver2/hy_furniture"
    source_files = (
        Path(__file__).resolve(),
        REPO_ROOT / "src/eval/evaluate_model.py",
        REPO_ROOT / "src/eval/rollout.py",
        REPO_ROOT / "src/eval/vlm_guidance.py",
        REPO_ROOT / "src/eval/vlm_point_metrics.py",
        REPO_ROOT / "src/vlm_data_generator.py",
        REPO_ROOT / "services/vlm_guidance/app.py",
        REPO_ROOT / "services/vlm_guidance/engine.py",
        REPO_ROOT / "services/vlm_guidance/native_sft.py",
    )
    manifest = {
        "version": 1,
        "stage": args.stage,
        "namespace": args.namespace,
        "created_at": _timestamp(),
        "n_envs": args.n_envs,
        "n_rollouts_per_cell": args.n_rollouts,
        "max_rollout_steps": args.max_rollout_steps,
        "max_saved_rollouts_per_cell": min(MAX_SAVED_ROLLOUTS, args.n_rollouts),
        "randomness": "low",
        "vlm_timeout_seconds": args.vlm_timeout_seconds,
        "vlm_query_interval": 8,
        "vlm_noise_projection_samples": 200,
        "total_requested_rollouts": len(rows) * args.n_rollouts,
        "vlm_readiness": readiness,
        "checkpoint_audits": audits,
        "robust_rearrangement": _git_provenance(REPO_ROOT),
        "gpu_snatcher": _git_provenance(args.auto_eval.parent),
        "auto_eval_sha256": _sha256(args.auto_eval),
        "hy_furniture": _git_provenance(hy_root),
        "hy_prediction_sha256": _sha256(hy_root / "prediction.py"),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_files
        },
        "data_dir_raw": str(args.data_dir_raw.resolve()),
        "runs": rows,
    }
    _atomic_json(manifest_path, manifest)
    print(f"validated {len(rows)} rows: {manifest_path}")
    return 0


def _run_phase(args) -> int:
    manifest_path = args.output_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = (args.stage, args.namespace, args.n_envs, args.n_rollouts, args.max_rollout_steps)
    actual = (
        manifest.get("stage"),
        manifest.get("namespace"),
        manifest.get("n_envs"),
        manifest.get("n_rollouts_per_cell"),
        manifest.get("max_rollout_steps"),
    )
    if actual != expected or any(row.get("status") != "pending" for row in manifest["runs"]):
        raise RuntimeError(f"manifest/runtime mismatch or already started: {actual} != {expected}")
    env = _child_env(args)
    for row in manifest["runs"]:
        before = _check_ready(args.vlm_base_url, os.environ.get("VLM_API_TOKEN"))
        gpu_free_before_mib = _gpu_memory_free_mib()
        row.update({"status": "running", "started_at": _timestamp(), "readiness_before": before})
        row["gpu_free_before_mib"] = gpu_free_before_mib
        _atomic_json(manifest_path, manifest)
        log_path = Path(row["stdout_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = row["auto_eval_command"]
        print(f"running {row['condition']}/{row['task']}: {shlex.join(command)}", flush=True)
        with log_path.open("w", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            captured = []
            for line in process.stdout:
                captured.append(line)
                sys.stdout.write(line)
                log.write(line)
            return_code = process.wait()
        output = "".join(captured)
        gpu_quiescence = _wait_for_gpu_quiescence(
            baseline_free_mib=gpu_free_before_mib
        )
        after = _check_ready(args.vlm_base_url, os.environ.get("VLM_API_TOKEN"))
        errors = [text for text in FORBIDDEN_LOG_TEXT if text in output]
        summary_path = Path(row["summary_path"])
        summary_error = None
        if not summary_path.is_file():
            summary_error = "summary missing"
        else:
            summary = json.loads(summary_path.read_text())
            if summary.get("n_rollouts") != args.n_rollouts:
                summary_error = f"n_rollouts={summary.get('n_rollouts')}"
            elif summary.get("vlm_model_revision") != EXPECTED_VLM_REVISION:
                summary_error = "VLM revision mismatch"
            elif not (((summary.get("vlm_point_error") or {}).get("all") or {}).get("overall") or {}).get("count_valid"):
                summary_error = "VLM point metrics empty"
        failures = []
        if return_code:
            failures.append(f"return_code={return_code}")
        failures.extend(errors)
        if summary_error:
            failures.append(summary_error)
        if before != after:
            failures.append("readiness changed")
        row.update(
            {
                "return_code": return_code,
                "finished_at": _timestamp(),
                "readiness_after": after,
                "gpu_quiescence_after": gpu_quiescence,
                "failure_reasons": failures,
                "status": "failed" if failures else "complete",
            }
        )
        _atomic_json(manifest_path, manifest)
        if failures:
            return 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("print", "run"), required=True)
    parser.add_argument("--stage", choices=("capacity", "smoke", "formal"), required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir-raw", type=Path, required=True)
    parser.add_argument(
        "--auto-eval",
        type=Path,
        default=Path("/home/huyue/projects/gpu-snatcher/auto_eval.sh"),
    )
    parser.add_argument(
        "--rr-python",
        type=Path,
        default=Path("/home/huyue/miniconda3/envs/rr/bin/python"),
    )
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int)
    parser.add_argument("--max-rollout-steps", type=int)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    defaults = {
        "capacity": (args.n_envs, 128),
        "smoke": (3, 1000),
        "formal": (36, 1000),
    }
    default_rollouts, default_steps = defaults[args.stage]
    args.n_rollouts = args.n_rollouts or default_rollouts
    args.max_rollout_steps = args.max_rollout_steps or default_steps
    if args.stage in {"smoke", "formal"} and (
        args.conditions != list(CONDITIONS) or args.tasks != list(TASKS)
    ):
        parser.error("smoke/formal require the complete fixed 2x3 matrix")
    if min(args.n_envs, args.n_rollouts, args.max_rollout_steps) <= 0:
        parser.error("rollout counts and max steps must be positive")
    return args


def main() -> int:
    args = parse_args()
    return _print_phase(args) if args.phase == "print" else _run_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
