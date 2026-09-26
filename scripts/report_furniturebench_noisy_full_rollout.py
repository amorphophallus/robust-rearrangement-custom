#!/usr/bin/env python3
"""Render auditable task and skill-level tables from full-rollout summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.eval.skill_level import INCLUDED_SKILL_STAGES


NOISE_LEVELS = ("n0", "n1", "n2", "n3", "n4", "n5", "n6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def read_summary_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "summaries").glob("*/*/*/*.json")):
        train_noise, train_seed, eval_noise, filename = path.relative_to(root / "summaries").parts
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "train_noise": train_noise,
            "train_seed": train_seed,
            "eval_noise": eval_noise,
            "task": filename[:-5] if filename.endswith(".json") else filename,
            "path": path,
            "summary": summary,
        })
    return rows


def sr_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    summary = row["summary"]
    return f"{summary['n_success']}/{summary['n_rollouts']} ({100 * summary['success_rate']:.1f}%)"


def aggregate_sr(rows: list[dict]) -> tuple[int, int, float] | None:
    if not rows:
        return None
    successes = sum(int(row["summary"]["n_success"]) for row in rows)
    rollouts = sum(int(row["summary"]["n_rollouts"]) for row in rows)
    return successes, rollouts, successes / rollouts


def aggregate_sr_cell(value: tuple[int, int, float] | None, *, best: bool) -> str:
    if value is None:
        return "—"
    successes, rollouts, rate = value
    display = f"{successes}/{rollouts} ({100 * rate:.1f}%)"
    return f"<strong>{display}</strong>" if best else display


def stage_cell(row: dict | None, stage: str, metric: str) -> str:
    if row is None:
        return "—"
    summary = row["summary"]
    reached = int(summary.get("skill_state_counts", {}).get(stage, 0))
    completed = int(summary.get("skill_completion_counts", {}).get(stage, 0))
    total = int(summary["n_rollouts"])
    if metric == "reached_total":
        return f"{reached}/{total} ({100 * reached / total:.1f}%)"
    if metric == "completed_reached":
        return "—" if reached == 0 else f"{completed}/{reached} ({100 * completed / reached:.1f}%)"
    if metric == "completed_total":
        return f"{completed}/{total} ({100 * completed / total:.1f}%)"
    raise ValueError(metric)


def main() -> int:
    args = parse_args()
    root = args.results_root.resolve()
    rows = read_summary_rows(root)
    index = {
        (row["train_noise"], row["train_seed"], row["eval_noise"], row["task"]): row
        for row in rows
    }
    conditions = sorted({(row["train_noise"], row["train_seed"]) for row in rows})
    complete = len(rows)
    # Pilot: 3 train conditions × 7 eval noise levels × 3 tasks.
    expected = 63 if root.name.startswith("full-rollout-pilot12") else None
    lines = [
        "# Noisy train/noisy eval: full-rollout pilot (12 rollouts per task)",
        "",
        "Each result cell is one independently persisted evaluator summary. "
        "The tables are regenerated directly from those JSON files; source paths "
        "are listed in the provenance appendix below.",
        "",
        f"- Result root: `{root}`",
        f"- Snapshot generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Complete task cells currently rendered: {complete}" + (f" / {expected}" if expected else ""),
        "- Runtime protocol: scripted annotations; low randomness; seed 0; 12 environments/12 rollouts; position tracking; physics-reward task success.",
        "- Evaluation noise is point-only clipped Gaussian (2σ), fixed within a semantic stage; base noise seed is recorded in every source JSON.",
        "",
    ]
    tasks = tuple(INCLUDED_SKILL_STAGES)
    overall: dict[tuple[str, str, str], tuple[int, int, float] | None] = {}
    for train_noise, train_seed in conditions:
        for level in NOISE_LEVELS:
            selected = [
                index[(train_noise, train_seed, level, task)]
                for task in tasks
                if (train_noise, train_seed, level, task) in index
            ]
            overall[(train_noise, train_seed, level)] = (
                aggregate_sr(selected) if len(selected) == len(tasks) else None
            )
        all_selected = [
            index[(train_noise, train_seed, level, task)]
            for level in NOISE_LEVELS
            for task in tasks
            if (train_noise, train_seed, level, task) in index
        ]
        overall[(train_noise, train_seed, "all")] = (
            aggregate_sr(all_selected)
            if len(all_selected) == len(NOISE_LEVELS) * len(tasks)
            else None
        )
    best_rates = {}
    for level in (*NOISE_LEVELS, "all"):
        values = [
            overall[(train_noise, train_seed, level)]
            for train_noise, train_seed in conditions
        ]
        best_rates[level] = max(
            (value[2] for value in values if value is not None), default=None
        )
    lines.extend([
        "## Overall task success rate",
        "",
        "每个 eval-noise 单元合并 `one_leg`、`round_table`、`lamp` 各 12 次，"
        "即 36 次 full rollout；`All eval` 再合并 N0–N6，共 252 次。"
        "三个任务样本数相同，因此 pooled SR 等于任务等权 macro average。"
        "每列粗体为最高值。",
        "",
        "| train checkpoint | " + " | ".join(level.upper() for level in NOISE_LEVELS) + " | All eval |",
        "|---|" + "|".join("---:" for _ in range(len(NOISE_LEVELS) + 1)) + "|",
    ])
    for train_noise, train_seed in conditions:
        values = []
        for level in (*NOISE_LEVELS, "all"):
            value = overall[(train_noise, train_seed, level)]
            best = (
                value is not None
                and best_rates[level] is not None
                and abs(value[2] - best_rates[level]) < 1e-12
            )
            values.append(aggregate_sr_cell(value, best=best))
        lines.append(f"| {train_noise}/{train_seed} | " + " | ".join(values) + " |")
    lines.append("")
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend([f"## {task}", "", "### Task success rate", ""])
        lines.append("| train checkpoint | " + " | ".join(NOISE_LEVELS) + " |")
        lines.append("|---|" + "|".join("---:" for _ in NOISE_LEVELS) + "|")
        for train_noise, train_seed in conditions:
            values = [sr_cell(index.get((train_noise, train_seed, level, task))) for level in NOISE_LEVELS]
            lines.append(f"| {train_noise}/{train_seed} | " + " | ".join(values) + " |")
        for metric, label in (
            ("reached_total", "Stage reached / all task rollouts"),
            ("completed_reached", "Current-stage completion / reached"),
            ("completed_total", "Stage completion / all task rollouts"),
        ):
            lines.extend(["", f"### {label}", ""])
            lines.append("| train / eval | " + " | ".join(stages) + " |")
            lines.append("|---|" + "|".join("---:" for _ in stages) + "|")
            for train_noise, train_seed in conditions:
                for level in NOISE_LEVELS:
                    row = index.get((train_noise, train_seed, level, task))
                    values = [stage_cell(row, stage, metric) for stage in stages]
                    lines.append(
                        f"| {train_noise}/{train_seed} → {level} | "
                        + " | ".join(values) + " |"
                    )
        lines.append("")
    lines.extend(["## Provenance appendix", ""])
    for row in rows:
        lines.append(
            f"- `{row['train_noise']}/{row['train_seed']} → {row['eval_noise']}/{row['task']}`: "
            f"`{row['path']}`"
        )
    output = args.markdown_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rendered={complete} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
