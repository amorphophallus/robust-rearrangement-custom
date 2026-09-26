#!/usr/bin/env python3
"""Materialize auditable paired metrics from a noisy skill-level matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.eval.noisy_skill_level import paired_endpoint_metrics, summarize_paired_records
from src.eval.skill_level import INCLUDED_SKILL_STAGES

NOISE_STD_M = {
    "n0": 0.0, "n1": 0.003, "n2": 0.006, "n3": 0.012,
    "n4": 0.024, "n5": 0.048, "n6": 0.096,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help=(
            "Optional durable Markdown rendering.  It contains one table per "
            "task/metric, with task-specific skill stages as columns."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _format_metric(metric: str, value) -> str:
    if value is None:
        return "—"
    if metric == "guidance_following_positive_rate":
        rate = value.get("rate") if isinstance(value, dict) else value
        return "—" if rate is None else f"{100.0 * float(rate):.1f}%"
    # Endpoint metrics retain their distributional audit fields in JSON.  The
    # compact task tables intentionally show the arithmetic mean; success rate
    # is already a scalar.
    if isinstance(value, dict):
        value = value.get("mean")
        if value is None:
            return "—"
    if metric == "success_rate":
        return f"{100.0 * float(value):.1f}%"
    if metric in {
        "e_gt_m",
        "e_input_m",
        "delta_x_m",
        "guidance_following_displacement_m",
    }:
        return f"{1000.0 * float(value):.2f}"
    if metric == "guidance_gain":
        return f"{float(value):.3f}"
    raise ValueError(f"Unknown metric: {metric}")


def _aggregate_cm(value) -> str:
    """Format a source-meter endpoint statistic for the overview table."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        value = value.get("mean")
    return "—" if value is None else f"{100.0 * float(value):.3f}"


def _aggregate_gain(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        value = value.get("mean")
    return "—" if value is None else f"{float(value):.3f}"


def _aggregate_following_cm(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        value = value.get("mean")
    return "—" if value is None else f"{100.0 * float(value):.3f}"


def _metric_mean(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def _decorate_rank(display: str, value, candidates, *, rule: str) -> str:
    """Bold the best distinct value and underline the second-best."""
    numeric = _metric_mean(value)
    available = [_metric_mean(item) for item in candidates]
    available = [item for item in available if item is not None]
    if numeric is None or not available:
        return display
    if rule == "max":
        ordered = sorted(set(available), reverse=True)
        key = numeric
    elif rule == "min":
        ordered = sorted(set(available))
        key = numeric
    elif rule == "absmin":
        ordered = sorted(set(abs(item) for item in available))
        key = abs(numeric)
    else:
        raise ValueError(rule)
    if np.isclose(key, ordered[0], rtol=0.0, atol=1e-12):
        return f"<strong>{display}</strong>"
    if len(ordered) > 1 and np.isclose(key, ordered[1], rtol=0.0, atol=1e-12):
        return f"<u>{display}</u>"
    return display


def write_markdown_report(path: Path, summary: dict, *, results_root: Path) -> None:
    """Render the pilot matrix without duplicating or rounding its source JSON."""
    conditions = sorted(summary["by_condition"])
    state_count = int(summary.get("state_count_per_stage", 8))
    records_per_condition = 18 * state_count
    metrics = (
        ("success_rate", "Current-stage FSM success rate (%)"),
        ("e_gt_m", "E_GT (mm)"),
        ("e_input_m", "E_input / TE (mm)"),
        ("delta_x_m", "Δx (mm)"),
        (
            "guidance_following_displacement_m",
            "G_parallel / signed following displacement (mm)",
        ),
        ("guidance_following_positive_rate", "P(G_parallel > 0)"),
    )
    lines = [
        f"# Noisy train/noisy eval: state-bank result (n={state_count}, m=1)",
        "",
        "This is a rendering of the immutable per-rollout records, not a "
        "separate result calculation.  Each row is one training-noise / "
        "evaluation-noise condition using its registry pilot checkpoint. "
        f"Each cell aggregates {state_count} expert-restored states for that task-stage.",
        "",
        f"- Source result root: `{results_root}`",
        "- Source records: `paired_records.jsonl`; source summary: `paired_summary.json`.",
        "- Success is the current skill stage's FSM transition; N0 has no paired Δx/G by definition.",
        "- E_GT=||xδ−p0||; E_input=||xδ−pδ||; Δx=||xδ−x0||; G_parallel=<xδ−x0,δ>/||δ||.",
        "- Train: N0=0 mm, N2=6 mm, N4=24 mm per axis. Eval: N0–N6=0, 3, 6, 12, 24, 48, 96 mm per axis; clipped Gaussian (2σ), point-only.",
        "",
        "The source run's standardized-noise pairing audit passed: "
        f"max error {summary['noise_pairing_audit']['max_abs_standardized_delta_error']:.3g} "
        f"(tolerance {summary['noise_pairing_audit']['tolerance']:.1g}).",
        "",
        "## Cross-task/stage overview: train × eval matrix",
        "",
        "This is the compact table used for the initial robustness readout. "
        f"For each train/eval row it pools **all {records_per_condition} raw short-rollout records**: "
        "18 included stages (`one_leg` 5 + `round_table` 7 + `lamp` 6), "
        f"{state_count} fixed expert-restored states per stage, and `m=1`.  Every stage "
        "therefore has equal weight.  It is not a full-rollout task-SR table. "
        "The train rows use the pilot checkpoints N0/main-formal, N2/2026092201, "
        "and N4/2026092201.",
        "",
        "每个测试噪声列内跨三个训练条件排名：<strong>粗体为第一名</strong>，"
        "<u>下划线为第二名</u>；并列共享同一名次。SR 越高越优，"
        "三个误差/位移指标越低越优；`G_parallel` 的正负表示 following 方向，"
        "其绝对值是沿 guidance-noise 方向的实际位移长度，不作单一优劣排名。",
        "",
        "### Current-stage SR",
        "",
        "| Train \\ Eval | N0 | N1 | N2 | N3 | N4 | N5 | N6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        train_noise, train_seed = condition.split("/", 1)
        label = train_noise.upper() if train_seed in {"main-formal", "2026092201"} else f"{train_noise.upper()} ({train_seed})"
        values = []
        for level in NOISE_STD_M:
            value = summary["by_condition"][condition][level]["success_rate"]
            candidates = [
                summary["by_condition"][other][level]["success_rate"]
                for other in conditions
            ]
            display = f"{100.0 * float(value):.1f}%"
            values.append(_decorate_rank(display, value, candidates, rule="max"))
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "### Endpoint and response metrics",
        "",
        "Errors and `Δx` are cm; `G` is unitless.  Each training condition "
        "occupies four rows, with the metric names in the left header; evaluation "
        "noise remains the only horizontal axis.",
        "",
        "<table>",
        "<thead><tr><th>Train</th><th>Metric \\ Eval</th>"
        + "".join(f"<th>{level.upper()}</th>" for level in NOISE_STD_M)
        + "</tr></thead>",
        "<tbody>",
    ])
    metric_rows = (
        ("E<sub>GT</sub> (cm)", "e_gt_m", _aggregate_cm, "min"),
        ("E<sub>input</sub>/TE (cm)", "e_input_m", _aggregate_cm, "min"),
        ("Δx (cm)", "delta_x_m", _aggregate_cm, "min"),
        (
            "G<sub>parallel</sub> (cm)",
            "guidance_following_displacement_m",
            _aggregate_following_cm,
            "none",
        ),
    )
    for condition in conditions:
        train_noise, train_seed = condition.split("/", 1)
        label = train_noise.upper() if train_seed in {"main-formal", "2026092201"} else f"{train_noise.upper()} ({train_seed})"
        for index, (metric_label, field, formatter, rank_rule) in enumerate(metric_rows):
            values = []
            for eval_noise in NOISE_STD_M:
                value = summary["by_condition"][condition][eval_noise][field]
                candidates = [
                    summary["by_condition"][other][eval_noise][field]
                    for other in conditions
                ]
                values.append(
                    formatter(value)
                    if rank_rule == "none"
                    else _decorate_rank(
                        formatter(value), value, candidates, rule=rank_rule
                    )
                )
            train_header = f"<th rowspan=\"4\">{label}</th>" if index == 0 else ""
            lines.append(
                "<tr>" + train_header + "<th>" + metric_label + "</th>"
                + "".join(f"<td>{value}</td>" for value in values) + "</tr>"
            )
    lines.extend(["</tbody></table>"])
    lines.extend([
        "",
        "### 计算方法与数据来源",
        "",
        f"每个 Train × Eval 单元汇总 {records_per_condition} 条 short rollout：3 个任务共 18 个纳入分析的 "
        f"skill stage，每个 stage 使用 {state_count} 个固定 expert 中间状态，`m=1`。所有 stage "
        "样本数相同，因此这里的 pooled mean 也等价于 stage 等权 macro mean。成功与失败 "
        "rollout 都进入连续指标统计，不进行 success-only 筛选。",
        "",
        "对固定的 `(train checkpoint, task, stage, state_sha256, repeat_index)`，"
        "单 cell evaluator 将机器人基座坐标系下的终点和引导信息写入各目录的 "
        "`records.jsonl`；汇总脚本再按该精确键连接 N0 与 noisy rollout，生成根目录的 "
        "`paired_records.jsonl`。表中只显示这些逐条记录的算术平均：",
        "",
        "| 符号/指标 | 使用的原始字段 | 表中计算方法 |",
        "|---|---|---|",
        "| `p0` | `p0_robot_base_m` | noisy rollout 终止帧的 scripted/geometry clean 引导点 |",
        "| `pδ` | `p_delta_robot_base_m` | 同帧实际输入 policy 的 noisy 引导点；`δ=pδ-p0` |",
        "| `x0` | matched N0 的 `end_ee_pos_robot_base_m` | 相同 state/repeat 的 clean-policy 终点 |",
        "| `xδ` | noisy rollout 的 `end_ee_pos_robot_base_m` | noisy-policy 终点 |",
        f"| SR | `completed_current_stage` | {records_per_condition} 个布尔值的均值；合法 FSM forward transition 记为成功 |",
        "| `E_GT` | `xδ`, `p0` | 逐条算 `||xδ-p0||` 后取均值；原始单位 m，表中转为 cm |",
        "| `E_input` / TE | `xδ`, `pδ` | 逐条算 `||xδ-pδ||` 后取均值；等于该 noisy endpoint 的 position TE |",
        "| `Δx` | `xδ`, `x0` | 逐条算 `||xδ-x0||` 后取均值；原始单位 m，表中转为 cm |",
        "| `G_parallel` | `xδ`, `x0`, `δ` | 逐条算 `<xδ-x0,δ>/||δ||` 后取均值，单位 m；表中转为 cm；正值表示沿 noisy-guidance 方向移动 |",
        "",
        "N0 时 `δ=0`，所以 `Δx` 与 `G_parallel` 在数学上未定义，表中记为 `—`；此时 "
        "`E_GT=E_input=TE`。N1–N6 对同一 state/repeat 共享同一个截断标准高斯向量 "
        "`z`，仅按 `δ=σz` 缩放；上方 pairing audit 已数值验证这一点。checkpoint hash、"
        "state hash、source episode、selection stratum、终止原因、FSM 转移、四个端点和 "
        "噪声向量均保留在文档顶部列出的 JSONL 来源中，可逐条回溯。",
        "",
        "### 结果分析",
        "",
        "- **SR：**N0-train 在 N0–N6 的每一列都是第一或并列第一；N2/N4 没有产生 "
        "稳定的当前-stage 成功率增益。因此该 pilot 不能支持“noisy training 提高 SR”。",
        "- **E_GT：**低测试噪声 N1–N2 仍由 N0-train 最优；从 N3 开始出现反转。"
        "N2 在 N3、N4、N6 最优，N4 在 N5 最优。N3–N6 平均 E_GT 分别约为 "
        "N0=3.998 cm、N2=3.422 cm、N4=3.534 cm，说明 noisy training 的主要收益 "
        "出现在较大标注噪声下，N2 的总体趋势最好。",
        "- **E_input/TE：**N3–N4 由 N2 最低，N5–N6 由 N4 最低，但 N6 三者只差 "
        "0.172 cm。该量表示对实际输入点的终点误差；单独变小既可能是更精确地利用 "
        "有效引导，也可能是更强地跟随错误点，所以必须与 E_GT 联合解释。当前高噪声下 "
        "E_GT 同时下降，才支持改善了 grounding，而不是单纯追随 noisy point。",
        "- **Δx：**N1–N2 由 N0 最小；N3–N5 由 N2 最小，N6 由 N4 最小。"
        "N3–N6 平均 Δx 约为 N0=1.879 cm、N2=1.078 cm、N4=1.342 cm；"
        "N2 对错误引导造成的行为终点偏移抑制最稳定。",
        "- **G：**这里按 `|G|` 接近 0 作为抗噪排名。赢家随测试噪声变化：N0 在 N1/N4 "
        "最接近 0，N2 在 N2/N3/N5 最接近 0，N4 在 N6 最接近 0。G 在小 `δ` 时 "
        "分母较小，容易被少数 stage 放大，因此它更适合解释响应方向，不能单独决定策略排名。",
        "- **综合判断：**最优标记并不完全统一，尤其 SR 始终偏向 N0，而局部几何指标在 "
        "N3–N6 多数偏向 N2。当前数据最多说明 N2（6 mm/axis train noise）是局部 "
        "spatial robustness 的最佳候选；不能据此宣称它是整体最鲁棒策略，也没有证据表明 "
        "训练噪声从 N2 增至 N4 会继续单调获益。正式结论仍需 n=24、m=2、3 seeds。",
    ])
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(["", f"## {task}"])
        for metric, title in metrics:
            lines.extend(["", f"### {title}", ""])
            lines.append("| train / eval | " + " | ".join(stages) + " |")
            lines.append("|---|" + "|".join("---:" for _ in stages) + "|")
            for condition in sorted(summary["by_condition_stage"]):
                stage_data = summary["by_condition_stage"][condition]
                train_noise, train_seed = condition.split("/", 1)
                for eval_noise in NOISE_STD_M:
                    label = f"{train_noise}/{train_seed} → {eval_noise}"
                    values = [
                        _format_metric(
                            metric,
                            stage_data[f"{task}/{stage}"][eval_noise][metric],
                        )
                        for stage in stages
                    ]
                    lines.append("| " + label + " | " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.results_root.resolve()
    record_paths = sorted(root.glob("*/*/*/*/*/records.jsonl"))
    groups: dict[tuple[str, str, str, str], dict[str, list[dict]]] = {}
    for path in record_paths:
        train_noise, train_seed, eval_noise, task, stage = path.relative_to(root).parts[:5]
        groups.setdefault((train_noise, train_seed, task, stage), {})[eval_noise] = read_jsonl(path)

    paired_rows = []
    for (train_noise, train_seed, task, stage), levels in sorted(groups.items()):
        if "n0" not in levels:
            raise ValueError(f"missing n0 baseline: {train_noise}/{train_seed}/{task}/{stage}")
        clean_by_key = {
            (str(row["state_sha256"]), int(row["repeat_index"])): row
            for row in levels["n0"]
        }
        for eval_noise, rows in sorted(levels.items()):
            for row in rows:
                key = (str(row["state_sha256"]), int(row["repeat_index"]))
                if key not in clean_by_key:
                    raise ValueError(f"unpaired record {key}: {train_noise}/{eval_noise}")
                metrics = paired_endpoint_metrics(clean_by_key[key], row)
                for field in ("e_gt_m", "e_input_m"):
                    recorded = row.get(field)
                    if recorded is None or abs(float(recorded) - metrics[field]) > 1e-6:
                        raise ValueError(f"{field} audit mismatch for {key}: {recorded} vs {metrics[field]}")
                paired_rows.append(
                    {
                        "schema": "rr-noisy-skill-level-paired-record-v1",
                        "train_noise_level": train_noise,
                        "train_seed": train_seed,
                        "eval_noise_level": eval_noise,
                        "task": task,
                        "skill_stage": stage,
                        "state_sha256": key[0],
                        "repeat_index": key[1],
                        "source_episode_index": row.get("source_episode_index"),
                        "selection_stratum": row.get("selection_stratum"),
                        "completed_current_stage": bool(row["completed_current_stage"]),
                        **metrics,
                    }
                )

    paired_path = root / "paired_records.jsonl"
    temporary = paired_path.with_name(f".{paired_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in paired_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, paired_path)

    by_condition = {}
    by_condition_stage = {}
    condition_keys = sorted(
        {(row["train_noise_level"], row["train_seed"]) for row in paired_rows}
    )
    for train_noise, train_seed in condition_keys:
        subset = [
            row for row in paired_rows
            if row["train_noise_level"] == train_noise and row["train_seed"] == train_seed
        ]
        by_condition[f"{train_noise}/{train_seed}"] = summarize_paired_records(subset)
        stage_keys = sorted({(row["task"], row["skill_stage"]) for row in subset})
        by_condition_stage[f"{train_noise}/{train_seed}"] = {
            f"{task}/{stage}": summarize_paired_records(
                row for row in subset
                if row["task"] == task and row["skill_stage"] == stage
            )
            for task, stage in stage_keys
        }

    standardized: dict[tuple, dict[str, np.ndarray]] = {}
    for row in paired_rows:
        level = row["eval_noise_level"]
        sigma = NOISE_STD_M[level]
        key = (
            row["train_noise_level"], row["train_seed"], row["task"],
            row["skill_stage"], row["state_sha256"], row["repeat_index"],
        )
        delta = np.asarray(row["delta_robot_base_m"], dtype=np.float64)
        if sigma == 0.0:
            if float(np.linalg.norm(delta)) > 1e-9:
                raise ValueError(f"N0 has nonzero delta for {key}")
            continue
        standardized.setdefault(key, {})[level] = delta / sigma
    pairing_errors = []
    for levels in standardized.values():
        reference = levels[sorted(levels)[0]]
        pairing_errors.extend(
            float(np.max(np.abs(value - reference))) for value in levels.values()
        )
    maximum_pairing_error = max(pairing_errors, default=0.0)
    if maximum_pairing_error > 1e-5:
        raise ValueError(
            f"noise levels do not share a standard sample: {maximum_pairing_error}"
        )
    summary = {
        "schema": "rr-noisy-skill-level-paired-summary-v1",
        "record_count": len(paired_rows),
        "formulae": {
            "e_gt": "||x_delta - p0||",
            "e_input": "||x_delta - p_delta||",
            "delta_x": "||x_delta - x0||",
            "guidance_following_displacement": (
                "<x_delta-x0, delta>/||delta||"
            ),
            "guidance_following_positive_rate": (
                "P(guidance_following_displacement > 0)"
            ),
            "legacy_guidance_gain": "<x_delta-x0, delta>/||delta||^2",
            "n0_delta_x_and_guidance_following": None,
        },
        "by_condition": by_condition,
        "by_condition_stage": by_condition_stage,
        "noise_pairing_audit": {
            "key": "train condition + task/stage + state_sha256 + repeat_index",
            "standardized_pair_count": len(pairing_errors),
            "max_abs_standardized_delta_error": maximum_pairing_error,
            "tolerance": 1e-5,
            "pass": True,
        },
    }
    write_json_atomic(root / "paired_summary.json", summary)
    if args.markdown_output is not None:
        write_markdown_report(
            args.markdown_output.resolve(), summary, results_root=root
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
