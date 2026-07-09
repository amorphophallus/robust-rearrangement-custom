#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# wandb test data is from last eval checkpoint (varies by run)
# absurd-voice-2 wandb data is from 2000-epoch run (3000 still training)
# For SR, use the matching epoch:
#   absurd-voice-2: 2000 SR (61.11%) since wandb data is from 2000 run
#   all others: 3000 SR since wandb data is from 3000 run
RUNS = [
    # (label, sr_3000, sr_2000, test_bc_loss, val_action_mse, use_2000_for_sr)
    ("icy-vortex-9\n(rgbd+skill)", 57.41, 45.37, 0.0920, 0.0266, False),
    ("clear-water-12\n(rgbd)", 13.89, 14.81, 0.1472, 0.0189, False),
    ("absurd-voice-2\n(colored GP)", 50.00, 50.00, 0.1186, 0.0471, True),
    ("rare-monkey-4\n(GP)", 52.78, 53.70, 0.1273, 0.0456, False),
    ("autumn-dust-13\n(GP)", 51.85, 55.56, 0.0879, 0.0269, False),
    ("fresh-tree-11\n(GP+skill)", 62.96, 52.78, 0.1009, 0.0322, False),
    ("true-firefly-8\n(rgb)", 5.56, 10.19, 0.2205, 0.0390, False),
]

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def default_output_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "reports" / "sr_vs_metrics.png"


def plot_sr_vs_metrics(output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for i, (label, sr3, sr2, loss, mse, use2k) in enumerate(RUNS):
        sr = sr2 if use2k else sr3
        ax.scatter(
            loss,
            sr,
            c=COLORS[i],
            s=100,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            label,
            (loss, sr),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7,
            color=COLORS[i],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Test BC Loss (wandb summary)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Success Rate vs Test BC Loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.07, 0.24)
    ax.set_ylim(0, 72)

    ax = axes[1]
    for i, (label, sr3, sr2, loss, mse, use2k) in enumerate(RUNS):
        sr = sr2 if use2k else sr3
        ax.scatter(
            mse,
            sr,
            c=COLORS[i],
            s=100,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            label,
            (mse, sr),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7,
            color=COLORS[i],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Val Action MSE Error (wandb summary)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Success Rate vs Val Action MSE Error", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.015, 0.050)
    ax.set_ylim(0, 72)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot success rate against wandb summary metrics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output image path. Defaults to reports/sr_vs_metrics.png in the repo root.",
    )
    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()
    plot_sr_vs_metrics(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
