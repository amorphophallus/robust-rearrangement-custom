"""Plot pooled skill-completion changes for the paper's Figure 2."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "data" / "skill_level_multiseed" / "cross_task_skill_type.csv"
STEM = HERE / "skill_level_condition_contrasts_multiseed"
SKILLS = ("push", "pick", "place", "screw")
CONDITIONS = (
    ("rgbd_skill", "Skill", "#DAB77D"),
    ("rgbd_gp", "GP", "#91B2C9"),
    ("rgbd_gp_skill", "GP+skill", "#689F8E"),
)


def load_rates() -> dict[tuple[str, str], float]:
    selected = {"rgbd", *(condition for condition, _, _ in CONDITIONS)}
    rates: dict[tuple[str, str], float] = {}
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            skill = row["skill_type"]
            condition = row["condition"]
            if skill not in SKILLS or condition not in selected:
                continue
            key = (condition, skill)
            if key in rates:
                raise ValueError(f"Duplicate source row: {key}")
            reached = int(row["reached_count"])
            completed = int(row["completed_count"])
            reported_rate = float(row["skill_success_rate"])
            if reached <= 0 or not np.isclose(reported_rate, completed / reached):
                raise ValueError(f"Inconsistent count/rate: {key}")
            rates[key] = reported_rate
    required = {(condition, skill) for condition in selected for skill in SKILLS}
    if rates.keys() != required:
        raise ValueError(f"Missing source rows: {required - rates.keys()}")
    return rates


def main() -> None:
    rates = load_rates()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
        }
    )
    figure = plt.figure(figsize=(7.2, 3.19))
    axis = figure.add_axes([0.10, 0.24, 0.87, 0.60])
    centers = np.arange(len(SKILLS), dtype=float)
    width = 0.205

    for index, (condition, label, color) in enumerate(CONDITIONS):
        values = [
            100 * (rates[condition, skill] - rates["rgbd", skill])
            for skill in SKILLS
        ]
        x_positions = centers + (index - 1) * width
        axis.bar(
            x_positions,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        for x, value in zip(x_positions, values):
            axis.text(
                x,
                value + (0.42 if value >= 0 else -0.42),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=6.7,
                color="#172C29",
            )

    axis.set_xlim(-0.55, 3.55)
    axis.set_xticks(centers, [skill.capitalize() for skill in SKILLS])
    axis.set_ylim(-15.5, 12.5)
    axis.set_yticks([-15, -10, -5, 0, 5, 10])
    axis.set_ylabel("Change in conditional completion vs RGB-D (pp)")
    axis.grid(axis="y", color="#E7E7E7", linewidth=0.5)
    axis.axhline(0, color="#42515A", linewidth=0.8, zorder=4)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", length=0, pad=7)
    figure.text(0.055, 0.945, "Skill-level changes relative to RGB-D", fontsize=10)
    figure.legend(
        handles=[Patch(facecolor=color, label=label) for _, label, color in CONDITIONS],
        loc="upper center",
        bbox_to_anchor=(0.77, 0.96),
        ncol=3,
        handlelength=1.2,
        columnspacing=1.5,
    )

    figure.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    figure.savefig(STEM.with_suffix(".svg"), facecolor="white")
    figure.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
