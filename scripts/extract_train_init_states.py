#!/usr/bin/env python3
"""Extract first-frame init states from an LMDB training dataset.

Usage:
    python extract_train_init_states.py <dataset.lmdb> [--output init_states.npz]

Output:
    A .npz file containing:
        tasks:          list of task label strings, one per episode
        parts_poses:    list of (n_parts*7+7,) numpy arrays (first frame)
        joint_positions: list of (7,) numpy arrays (first frame, or default if unavailable)
        gripper_finger_1: list of floats
        gripper_finger_2: list of floats
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.lmdb import (
    open_lmdb_env,
    read_lmdb_meta,
    read_lmdb_episode_index,
    episode_data_key,
    unpack_named_arrays,
)

# Default Franka joint positions (neutral pose for furniture assembly tasks)
DEFAULT_JOINT_POSITIONS = np.array(
    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64
)
DEFAULT_GRIPPER_FINGER_1 = 0.04
DEFAULT_GRIPPER_FINGER_2 = 0.04


def extract_joint_positions(episode_arrays, meta, frame_idx=0):
    """Extract joint_positions from episode data if available.

    Returns (joint_positions, gripper_finger_1, gripper_finger_2) or defaults.
    """
    robot_state = episode_arrays.get("robot_state")
    if robot_state is None:
        return DEFAULT_JOINT_POSITIONS.copy(), DEFAULT_GRIPPER_FINGER_1, DEFAULT_GRIPPER_FINGER_2

    frame_data = robot_state[frame_idx]

    lowdim_specs = meta.get("lowdim_specs", {})
    robot_spec = lowdim_specs.get("robot_state", {})

    if not robot_spec:
        # Without specs, try common dimension sizes
        if len(frame_data) >= 16:
            # Heuristic: robot_state = [ee_pos(3), ee_rot_6d(6), gripper_width(1),
            #                           joint_pos(7)?, ...]
            # The exact layout depends on the training config.
            pass
        return DEFAULT_JOINT_POSITIONS.copy(), DEFAULT_GRIPPER_FINGER_1, DEFAULT_GRIPPER_FINGER_2

    # Use specs to find joint_positions offset
    offset = robot_spec.get("offset", 0)
    shape = robot_spec.get("shape", [len(frame_data)])
    dtype_str = robot_spec.get("dtype", "float64")
    nbytes = robot_spec.get("nbytes", 0)

    # If the spec has a 2D shape, first dim is time
    if len(shape) >= 2:
        dim = shape[1]
    else:
        dim = shape[0]

    # Try to find joint_positions in lowdim_specs
    for key in ["joint_positions", "robot_state/joint_positions"]:
        if key in lowdim_specs:
            spec = lowdim_specs[key]
            jp_offset = spec.get("offset", 0)
            jp_shape = spec.get("shape", [7])
            jp_dim = jp_shape[-1] if isinstance(jp_shape, list) else jp_shape
            jp_dtype = spec.get("dtype", "float64")
            jp_nbytes = spec.get("nbytes", jp_dim * 8)

            # Concatenated: episode_data["robot_state"] may be a different key
            # Try extracting from the flat robot_state
            if jp_offset + jp_nbytes <= robot_state.shape[-1] * 8:
                return (
                    robot_state[frame_idx, jp_offset // 8 : (jp_offset + jp_nbytes) // 8].copy(),
                    DEFAULT_GRIPPER_FINGER_1,
                    DEFAULT_GRIPPER_FINGER_2,
                )

    # Fallback: return default joint positions
    return DEFAULT_JOINT_POSITIONS.copy(), DEFAULT_GRIPPER_FINGER_1, DEFAULT_GRIPPER_FINGER_2


def main():
    parser = argparse.ArgumentParser(
        description="Extract first-frame init states from LMDB training dataset"
    )
    parser.add_argument("dataset", type=str, help="Path to the .lmdb dataset directory")
    parser.add_argument(
        "--output", "-o", type=str, default="train_init_states.npz",
        help="Output .npz file path (default: train_init_states.npz)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_dir():
        print(f"Error: dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading metadata from {dataset_path}...")
    meta = read_lmdb_meta(dataset_path)
    episode_index = read_lmdb_episode_index(dataset_path)
    n_episodes = len(episode_index)
    print(f"Found {n_episodes} episodes")

    tasks = []
    parts_poses_list = []
    joint_positions_list = []
    gripper_f1_list = []
    gripper_f2_list = []

    env = open_lmdb_env(dataset_path, readonly=True)
    try:
        with env.begin(write=False) as txn:
            for ep_idx, ep_meta in enumerate(episode_index):
                task = ep_meta.get("task", "unknown")
                tasks.append(task)

                # Load episode data
                raw_episode = txn.get(episode_data_key(ep_idx))
                if raw_episode is None:
                    print(f"  Warning: episode {ep_idx} has no data, skipping...", file=sys.stderr)
                    parts_poses_list.append(np.zeros(7))
                    joint_positions_list.append(DEFAULT_JOINT_POSITIONS.copy())
                    gripper_f1_list.append(DEFAULT_GRIPPER_FINGER_1)
                    gripper_f2_list.append(DEFAULT_GRIPPER_FINGER_2)
                    continue

                ep_arrays = unpack_named_arrays(raw_episode)

                # Extract parts_poses (first frame)
                if "parts_poses" in ep_arrays:
                    parts_poses = ep_arrays["parts_poses"][0].copy()
                else:
                    print(
                        f"  Warning: episode {ep_idx} ({task}) has no parts_poses, using zeros",
                        file=sys.stderr,
                    )
                    parts_poses = np.zeros(7)
                parts_poses_list.append(parts_poses)

                # Extract joint positions
                jp, gf1, gf2 = extract_joint_positions(ep_arrays, meta, frame_idx=0)
                joint_positions_list.append(jp)
                gripper_f1_list.append(gf1)
                gripper_f2_list.append(gf2)

    finally:
        env.close()

    # Save to npz
    output_path = Path(args.output)
    print(f"Saving {n_episodes} init states to {output_path}...")

    # Convert lists to object arrays for npz storage
    np.savez_compressed(
        output_path,
        tasks=np.array(tasks, dtype=object),
        parts_poses=np.array(parts_poses_list, dtype=object),
        joint_positions=np.array(joint_positions_list, dtype=object),
        gripper_finger_1=np.array(gripper_f1_list),
        gripper_finger_2=np.array(gripper_f2_list),
    )

    # Print per-task summary
    task_counts = {}
    for t in tasks:
        task_counts[t] = task_counts.get(t, 0) + 1
    print("Per-task episode counts:")
    for t, c in sorted(task_counts.items()):
        print(f"  {t}: {c}")
    print("Done!")


if __name__ == "__main__":
    main()
