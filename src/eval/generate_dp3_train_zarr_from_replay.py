import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import zarr
from numcodecs import Pickle
import torch
from tqdm.auto import tqdm

from src.eval.replay import main as replay_main


def run_replay_for_dir(src_dir: Path, out_dir: Path, task: str, gpu: int, num_envs: int,
                       pc_points: int, headless: bool, visualize: bool) -> List[Path]:
    """Replay all pickles in src_dir, saving DP3-format pickles to out_dir. Returns list of output paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    pickles = sorted(src_dir.glob("*.pkl"))
    total = len(pickles)
    success_cnt = 0
    for pkl in tqdm(pickles, desc="replay", unit="traj"):
        args_list = [
            "--pickle-path", str(pkl),
            "--task", task,
            "--gpu", str(gpu),
            "--num-envs", str(num_envs),
            "--pc-points", str(pc_points),
            "--pc-out-dir", str(out_dir),
            "--action_type", "pos",
            "--headless" if headless else "--visualize",
        ]
        if visualize:
            args_list.append("--visualize")
        # Enable point cloud saving
        args_list.append("--save-pc-for-dp3")
        # Filter empty strings
        args_list = [a for a in args_list if a]
        replay_main(args_list)
        out_path = out_dir / pkl.name
        if out_path.exists():
            generated.append(out_path)
            success_cnt += 1
        tqdm.write(f"success {success_cnt}/{total} | replayed {len(generated)}/{total}")
    return generated


def merge_pickles_to_zarr(pickle_paths: List[Path], zarr_path: Path):
    """Merge multiple DP3 pickles into a single zarr with keys ['state','action','point_cloud','img'].
    Uses object arrays to store variable-length point clouds.
    """
    trajs = []
    for p in pickle_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        trajs.append(data)

    # Flatten all trajectories into aligned lists
    states, actions, pcls, imgs, idx = [], [], [], [], []
    for tid, t in enumerate(trajs):
        T = len(t["action"])
        states.extend(t["state"])
        actions.extend(t["action"])
        pcls.extend(t["point_cloud"])
        imgs.extend(t["img"])
        idx.extend([(tid, step) for step in range(T)])

    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_dataset("trajectory_index", data=np.asarray(idx, dtype=np.int32))
    root.create_dataset("state", data=np.asarray(states, dtype=object), object_codec=Pickle())
    root.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
    root.create_dataset("point_cloud", data=np.asarray(pcls, dtype=object), object_codec=Pickle())
    root.create_dataset("img", data=np.asarray(imgs, dtype=object), object_codec=Pickle())


def build_sampler(zarr_path: Path):
    """Return a simple sampler callable that yields (state, action, point_cloud, img) by trajectory chunks."""
    root = zarr.open_group(str(zarr_path), mode="r")
    idx = root["trajectory_index"][:]
    states = root["state"]
    actions = root["action"]
    pcls = root["point_cloud"]
    imgs = root["img"]

    # Precompute start/end per trajectory
    traj_to_range = {}
    for i, (tid, step) in enumerate(idx):
        if tid not in traj_to_range:
            traj_to_range[tid] = [i, i]
        traj_to_range[tid][1] = i

    def sampler():
        for tid, (s, e) in traj_to_range.items():
            sl = slice(s, e + 1)
            yield {
                "trajectory_id": tid,
                "state": states[sl],
                "action": actions[sl],
                "point_cloud": pcls[sl],
                "img": imgs[sl],
            }

    return sampler


def main():
    parser = argparse.ArgumentParser(description="Generate DP3 training data from replays")
    parser.add_argument("--src-dir", type=str, required=True, help="Input pickles to replay")
    parser.add_argument("--dp3-out-dir", type=str, required=True, help="Dir to store DP3 pickles with point clouds")
    parser.add_argument("--zarr-path", type=str, required=True, help="Output zarr path for merged dataset")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--pc-points", type=int, default=4096)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--only-replay", action="store_true", help="Only run replay to generate DP3 pickles")
    parser.add_argument("--only-merge", action="store_true", help="Only merge existing DP3 pickles to zarr")

    args = parser.parse_args()

    do_replay = not args.only_merge
    do_merge = not args.only_replay

    generated = []
    if do_replay:
        generated = run_replay_for_dir(
            src_dir=Path(args.src_dir),
            out_dir=Path(args.dp3_out_dir),
            task=args.task,
            gpu=args.gpu,
            num_envs=args.num_envs,
            pc_points=args.pc_points,
            headless=args.headless,
            visualize=args.visualize,
        )

    if do_merge:
        pickles = generated if generated else sorted(Path(args.dp3_out_dir).glob("*.pkl"))
        merge_pickles_to_zarr(pickles, Path(args.zarr_path))


if __name__ == "__main__":
    main()
