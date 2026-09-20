"""Build a qualitative front-camera sequence from one physical rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


FRAMES = (
    (60, (496, 201), "Part target", "#00bfc5"),
    (130, (581, 302), "Assembly target", "#dc424b"),
    (320, (523, 263), "Contact approach", "#dc424b"),
)
HEADER_CROP = 35
BOTTOM_CROP = 10


def build(video_path: Path, output_dir: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    fps = float(metadata["fps"])
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    figure.subplots_adjust(left=0.022, right=0.985, top=0.84, bottom=0.16, wspace=0.038)

    for panel_index, (axis, (frame_index, point, label, accent)) in enumerate(
        zip(axes, FRAMES)
    ):
        frame = reader.get_data(frame_index)
        if frame.shape[:2] != (960, 1280):
            raise ValueError(f"Unexpected video frame size: {frame.shape[:2]}")
        front = frame[:480, :640]
        scene = front[HEADER_CROP : 480 - BOTTOM_CROP]
        px, py = point[0], point[1] - HEADER_CROP

        axis.imshow(scene, interpolation="nearest")
        axis.set_axis_off()
        axis.add_patch(
            Circle((px, py), radius=11, fill=False, edgecolor="black", linewidth=2.0)
        )
        axis.add_patch(
            Circle((px, py), radius=11, fill=False, edgecolor="white", linewidth=1.15)
        )

        half_window = 23
        enlarged = scene[
            py - half_window : py + half_window,
            px - half_window : px + half_window,
        ]
        inset = axis.inset_axes((0.035, 0.64, 0.31, 0.34))
        inset.imshow(enlarged, interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(1.25)
            spine.set_edgecolor(accent)

        axis.text(
            0,
            1.035,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#202632",
        )
        axis.text(
            0.085,
            1.035,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#202632",
        )
        axis.text(
            1,
            1.035,
            f"{frame_index / fps:.0f} s",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#5b6571",
        )

    reader.close()

    figure.text(0.025, 0.075, "●", fontsize=9, color="#00bfc5", va="center")
    figure.text(0.049, 0.075, "Pick / Screw", fontsize=7, va="center", color="#202632")
    figure.text(0.214, 0.075, "●", fontsize=9, color="#dc424b", va="center")
    figure.text(
        0.238,
        0.075,
        "Place / Push / Insert",
        fontsize=7,
        va="center",
        color="#202632",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "real_one_leg_guidance_sequence"
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    build(arguments.video, arguments.output_dir)


if __name__ == "__main__":
    main()
