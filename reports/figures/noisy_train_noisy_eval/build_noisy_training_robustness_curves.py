#!/usr/bin/env python3
"""Build the compact noisy-train/noisy-eval robustness figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOISE_LEVELS = ("n0", "n1", "n2", "n3", "n4", "n5", "n6")
NOISE_MM = (0, 3, 6, 12, 24, 48, 96)
TASKS = ("one_leg", "round_table", "lamp")
CONDITIONS = (
    ("n0", "main-formal", "Train N0", "#4C78A8"),
    ("n2", "2026092201", "Train N2", "#F58518"),
    ("n4", "2026092201", "Train N4", "#54A24B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--state-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def load_full_rollout(root: Path) -> tuple[dict, list[dict]]:
    data: dict = {}
    sources = []
    for train_noise, train_seed, label, _ in CONDITIONS:
        data[label] = {}
        for level in NOISE_LEVELS:
            data[label][level] = {}
            for task in TASKS:
                path = root / "summaries" / train_noise / train_seed / level / f"{task}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                data[label][level][task] = {
                    "n_success": int(row["n_success"]),
                    "n_rollouts": int(row["n_rollouts"]),
                    "success_rate": float(row["success_rate"]),
                }
                sources.append({"path": display_path(path), "sha256": sha256(path)})
    return data, sources


def full_overall(data: dict, label: str, level: str) -> float:
    rows = data[label][level].values()
    return sum(row["n_success"] for row in rows) / sum(
        row["n_rollouts"] for row in rows
    )


def state_mean(summary: dict, condition: str, level: str, metric: str) -> float:
    value = summary["by_condition"][condition][level][metric]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def setup_axis(ax, title: str, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Eval noise / σ (mm per axis)")
    ax.set_xticks(range(len(NOISE_LEVELS)))
    ax.set_xticklabels(
        [f"{level.upper()}\n{mm}" for level, mm in zip(NOISE_LEVELS, NOISE_MM)]
    )
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    args = parse_args()
    full_root = args.full_root.resolve()
    state_path = args.state_summary.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    full, full_sources = load_full_rollout(full_root)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    x = np.arange(len(NOISE_LEVELS))
    plotted: dict[str, dict[str, list[float | None]]] = {
        "full_rollout_overall_sr_percent": {},
        "state_bank_e_gt_cm": {},
        "state_bank_delta_x_cm": {},
    }

    fig, axes = plt.subplots(
        1, 3, figsize=(15.2, 4.7), constrained_layout=True, sharex=True
    )
    selected = (
        ("full_sr", "A. End-to-end task success", "Overall task SR (%)", (30, 85)),
        ("e_gt_m", "B. Clean-target accuracy", "E_GT (cm; lower is better)", (3.1, 4.7)),
        ("delta_x_m", "C. Behavior sensitivity", "Δx (cm; lower is better)", (0, 2.7)),
    )
    payload_keys = (
        "full_rollout_overall_sr_percent",
        "state_bank_e_gt_cm",
        "state_bank_delta_x_cm",
    )
    for ax, (metric, title, ylabel, ylim), payload_key in zip(
        axes, selected, payload_keys
    ):
        ax.axvspan(2.5, 6.5, color="#EEEEEE", alpha=0.7)
        for train_noise, train_seed, label, color in CONDITIONS:
            condition = f"{train_noise}/{train_seed}"
            if metric == "full_sr":
                values = [
                    100.0 * full_overall(full, label, level)
                    for level in NOISE_LEVELS
                ]
            else:
                values = []
                for level in NOISE_LEVELS:
                    raw = state["by_condition"][condition][level][metric]
                    if isinstance(raw, dict):
                        raw = raw.get("mean")
                    values.append(None if raw is None else 100.0 * float(raw))
            plotted[payload_key][label] = values
            ax.plot(
                x,
                [np.nan if value is None else value for value in values],
                marker="o",
                linewidth=2.3,
                markersize=5.4,
                label=label,
                color=color,
            )
        setup_axis(ax, title, ylabel, ylim)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Noisy training improves robustness to corrupted guidance",
        fontsize=14,
        fontweight="semibold",
    )

    stem = "noisy_training_robustness_curves"
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    payload = {
        "schema": "rr-noisy-training-robustness-curves-v1",
        "noise_levels": list(NOISE_LEVELS),
        "noise_std_mm_per_axis": list(NOISE_MM),
        "high_noise_levels": ["n3", "n4", "n5", "n6"],
        "plotted_values": plotted,
        "sources": {
            "full_rollout_summaries": full_sources,
            "state_bank_summary": {
                "path": display_path(state_path),
                "sha256": sha256(state_path),
            },
        },
    }
    (output / f"{stem}_data.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "stem": stem}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
