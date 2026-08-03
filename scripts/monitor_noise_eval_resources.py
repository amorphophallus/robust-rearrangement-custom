from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


GIB = 1024**3


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_pressure(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        values: dict[str, float] = {}
        for field in fields[1:]:
            key, _, value = field.partition("=")
            try:
                values[key] = float(value)
            except ValueError:
                continue
        result[fields[0]] = values
    return result


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, raw_value = line.partition(":")
        fields = raw_value.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values


def _run(
    command: list[str], timeout: int = 30, *, output_from_start: bool = False
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "output": (
                completed.stdout[:20_000]
                if output_from_start
                else completed.stdout[-20_000:]
            ),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "output": str(exc)}


def _service_properties(unit: str) -> dict[str, str]:
    result = _run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
            "--property=NRestarts",
            "--property=ControlGroup",
        ]
    )
    properties: dict[str, str] = {}
    if result["returncode"] == 0:
        for line in result["output"].splitlines():
            key, separator, value = line.partition("=")
            if separator:
                properties[key] = value
    return properties


def _cgroup_stats(cgroup_root: Path, relative_path: str) -> dict[str, Any]:
    if not relative_path:
        return {}
    path = cgroup_root / relative_path.lstrip("/")
    return {
        "path": str(path),
        "memory_current_bytes": _read_int(path / "memory.current"),
        "memory_peak_bytes": _read_int(path / "memory.peak"),
        "memory_swap_current_bytes": _read_int(path / "memory.swap.current"),
        "pids_current": _read_int(path / "pids.current"),
        "pressure": _read_pressure(path / "memory.pressure"),
    }


def _sample(args: argparse.Namespace) -> dict[str, Any]:
    memory = _meminfo()
    swap_total = memory.get("SwapTotal", 0)
    swap_free = memory.get("SwapFree", 0)
    service = _service_properties(args.unit)
    disk = shutil.disk_usage(args.repo_root)
    uid = os.getuid()
    user_cgroup = f"/user.slice/user-{uid}.slice/user@{uid}.service"
    return {
        "timestamp": _now(),
        "service": service,
        "memory": {
            "total_bytes": memory.get("MemTotal", 0),
            "available_bytes": memory.get("MemAvailable", 0),
            "swap_total_bytes": swap_total,
            "swap_used_bytes": max(0, swap_total - swap_free),
            "swap_used_ratio": (
                max(0, swap_total - swap_free) / swap_total if swap_total else 0.0
            ),
        },
        "system_pressure": _read_pressure(Path("/proc/pressure/memory")),
        "user_cgroup": _cgroup_stats(args.cgroup_root, user_cgroup),
        "service_cgroup": _cgroup_stats(
            args.cgroup_root, service.get("ControlGroup", "")
        ),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def _pressure_avg10(sample: dict[str, Any], scope: str) -> float:
    pressure = (
        sample.get(scope, {}).get("pressure", {})
        if scope.endswith("cgroup")
        else sample.get(scope, {})
    )
    return float(pressure.get("full", {}).get("avg10", 0.0))


def _classify(sample: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    available_gib = sample["memory"]["available_bytes"] / GIB
    swap_ratio = sample["memory"]["swap_used_ratio"]
    disk_free_gib = sample["disk"]["free_bytes"] / GIB
    system_psi = _pressure_avg10(sample, "system_pressure")
    user_psi = _pressure_avg10(sample, "user_cgroup")

    critical: list[str] = []
    warnings: list[str] = []
    if available_gib <= args.critical_available_gib:
        critical.append(f"MemAvailable={available_gib:.2f}GiB")
    elif available_gib <= args.warn_available_gib:
        warnings.append(f"MemAvailable={available_gib:.2f}GiB")
    if max(system_psi, user_psi) >= args.critical_psi:
        critical.append(f"memory.full.avg10={max(system_psi, user_psi):.2f}%")
    elif max(system_psi, user_psi) >= args.warn_psi:
        warnings.append(f"memory.full.avg10={max(system_psi, user_psi):.2f}%")
    if swap_ratio >= args.critical_swap_ratio and available_gib <= args.warn_available_gib:
        critical.append(f"swap_used={swap_ratio:.1%}")
    elif swap_ratio >= args.warn_swap_ratio:
        warnings.append(f"swap_used={swap_ratio:.1%}")
    if disk_free_gib <= args.critical_disk_free_gib:
        critical.append(f"disk_free={disk_free_gib:.1f}GiB")
    elif disk_free_gib <= args.warn_disk_free_gib:
        warnings.append(f"disk_free={disk_free_gib:.1f}GiB")

    if critical:
        return "critical", critical + warnings
    if warnings:
        return "warning", warnings
    return "ok", []


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _diagnose(args: argparse.Namespace, sample: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    since = "10 minutes ago"
    diagnostics = {
        "timestamp": _now(),
        "severity": sample["severity"],
        "reasons": reasons,
        "sample": sample,
        "commands": [
            _run(["systemctl", "--user", "status", args.unit, "--no-pager"]),
            _run(
                ["ps", "-eo", "user,pid,ppid,rss,%mem,etimes,cmd", "--sort=-rss"],
                output_from_start=True,
            ),
            _run(["oomctl", "dump"]),
            _run(["nvidia-smi"]),
            _run(
                [
                    "journalctl",
                    "--since",
                    since,
                    "--no-pager",
                    "-o",
                    "short-precise",
                ]
            ),
        ],
    }
    if args.experiment_log.exists():
        diagnostics["experiment_log_tail"] = args.experiment_log.read_text(
            errors="replace"
        )[-20_000:]
    _append_jsonl(args.alerts_log, diagnostics)
    return diagnostics


def _restart(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    result = _run(["systemctl", "--user", "restart", args.unit], timeout=60)
    state["auto_restart_count"] += 1
    state["last_restart_at"] = _now()
    state["last_restart_result"] = result
    return result


def _audit_results(args: argparse.Namespace) -> dict[str, Any]:
    result = _run(
        [
            str(args.conda),
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            "-m",
            "scripts.audit_clean_train_noise_eval",
            "--manifest",
            str(args.manifest),
        ],
        timeout=300,
    )
    audit: dict[str, Any] = {"timestamp": _now(), **result}
    for line in reversed(result["output"].splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            audit["payload"] = payload
            break
    _append_jsonl(args.audit_log, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-log", type=Path, required=True)
    parser.add_argument("--samples-log", type=Path, required=True)
    parser.add_argument("--alerts-log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--audit-interval", type=float, default=3600.0)
    parser.add_argument(
        "--conda", type=Path, default=Path.home() / "miniconda3" / "bin" / "conda"
    )
    parser.add_argument("--conda-env", default="rr")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--warn-available-gib", type=float, default=6.0)
    parser.add_argument("--critical-available-gib", type=float, default=3.0)
    parser.add_argument("--warn-psi", type=float, default=20.0)
    parser.add_argument("--critical-psi", type=float, default=35.0)
    parser.add_argument("--warn-swap-ratio", type=float, default=0.90)
    parser.add_argument("--critical-swap-ratio", type=float, default=0.97)
    parser.add_argument("--warn-disk-free-gib", type=float, default=100.0)
    parser.add_argument("--critical-disk-free-gib", type=float, default=50.0)
    parser.add_argument("--critical-samples", type=int, default=2)
    parser.add_argument("--diagnostic-cooldown", type=float, default=300.0)
    parser.add_argument("--restart-cooldown", type=float, default=900.0)
    parser.add_argument("--max-auto-restarts", type=int, default=3)
    parser.add_argument("--auto-restart", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    state: dict[str, Any] = {
        "started_at": _now(),
        "last_sample_at": None,
        "last_severity": "unknown",
        "last_reasons": [],
        "last_diagnostic_at": None,
        "last_restart_at": None,
        "last_restart_result": None,
        "last_audit_at": None,
        "last_audit": None,
        "auto_restart_count": 0,
        "critical_streak": 0,
        "needs_attention": False,
    }
    last_diagnostic_monotonic = -float("inf")
    last_restart_monotonic = -float("inf")
    last_audit_monotonic = -float("inf")

    while True:
        sample = _sample(args)
        severity, reasons = _classify(sample, args)
        sample["severity"] = severity
        sample["reasons"] = reasons
        _append_jsonl(args.samples_log, sample)

        state["last_sample_at"] = sample["timestamp"]
        state["last_severity"] = severity
        state["last_reasons"] = reasons
        state["critical_streak"] = (
            state["critical_streak"] + 1 if severity == "critical" else 0
        )

        now_monotonic = time.monotonic()
        if now_monotonic - last_audit_monotonic >= args.audit_interval:
            audit = _audit_results(args)
            state["last_audit_at"] = audit["timestamp"]
            state["last_audit"] = audit.get("payload", audit)
            last_audit_monotonic = now_monotonic
            payload = audit.get("payload", {})
            if audit.get("returncode") != 0 or payload.get("issues"):
                _append_jsonl(
                    args.alerts_log,
                    {
                        "timestamp": _now(),
                        "event": "result_audit_failed",
                        "audit": audit,
                    },
                )

        severity_changed = severity != state.get("previous_severity")
        if severity != "ok" and (
            severity_changed
            or now_monotonic - last_diagnostic_monotonic >= args.diagnostic_cooldown
        ):
            _diagnose(args, sample, reasons)
            state["last_diagnostic_at"] = _now()
            last_diagnostic_monotonic = now_monotonic

        should_restart = (
            args.auto_restart
            and severity == "critical"
            and state["critical_streak"] >= args.critical_samples
            and state["auto_restart_count"] < args.max_auto_restarts
            and now_monotonic - last_restart_monotonic >= args.restart_cooldown
            and sample["service"].get("ActiveState") in {"active", "activating", "failed"}
        )
        if should_restart:
            _restart(args, state)
            last_restart_monotonic = now_monotonic
            state["critical_streak"] = 0
        elif (
            severity == "critical"
            and state["critical_streak"] >= args.critical_samples
            and state["auto_restart_count"] >= args.max_auto_restarts
        ):
            state["needs_attention"] = True

        state["previous_severity"] = severity
        _write_status(args.status, state)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
