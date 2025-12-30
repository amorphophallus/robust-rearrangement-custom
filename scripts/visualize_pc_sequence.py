import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def load_point_cloud_sequence(pickle_path: Path, max_points: Optional[int] = None) -> np.ndarray:
    """Load a (T, N, 3) point cloud sequence from pickle and optionally subsample."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "point_cloud" not in data:
        raise KeyError("Expected 'point_cloud' key in pickle data")

    seq = np.asarray(data["point_cloud"], dtype=np.float32)
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"point_cloud must have shape (T, N, 3); got {seq.shape}")

    print(f"Loaded point_cloud with shape {seq.shape} from {pickle_path}")

    if max_points is not None and seq.shape[1] > max_points:
        idx = np.random.permutation(seq.shape[1])[:max_points]
        seq = seq[:, idx, :]
        print(f"Subsampled to max_points={max_points}; new shape {seq.shape}")

    # Report basic stats for debugging
    nonempty = (np.linalg.norm(seq, axis=-1) > 0).sum(axis=1)
    print(
        "Frame nonzero counts (min/median/max):",
        int(nonempty.min()),
        int(np.median(nonempty)),
        int(nonempty.max()),
    )

    return seq


def visualize_sequence(seq: np.ndarray, fps: float, loop: bool):
    """Display point cloud frames in Open3D with simple playback."""
    if seq.shape[1] == 0:
        raise ValueError("Point cloud sequence has zero points; nothing to visualize")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence", width=960, height=720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seq[0])
    # Use a single color to keep rendering light
    pcd.paint_uniform_color([0.2, 0.6, 1.0])
    vis.add_geometry(pcd)

    period = 1.0 / max(fps, 1e-3)

    try:
        while True:
            for frame in seq:
                pcd.points = o3d.utility.Vector3dVector(frame)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(period)
            if not loop:
                break
    finally:
        vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a (T, N, 3) point cloud sequence from pickle using Open3D.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/rollout/low/pc/success/2025-12-14T17:33:11.871804.pkl"
        ),
        help="Path to pickle with point_cloud key (shape: T x N x 3)",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Playback speed in frames per second")
    parser.add_argument("--max-points", type=int, default=None, help="Optional random subsample of points per frame")
    parser.add_argument("--loop", action="store_true", help="Loop the sequence continuously")
    return parser.parse_args()


def main():
    args = parse_args()
    seq = load_point_cloud_sequence(args.input, max_points=args.max_points)
    visualize_sequence(seq, fps=args.fps, loop=args.loop)


if __name__ == "__main__":
    main()
