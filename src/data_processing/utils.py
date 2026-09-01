import pickle
import tarfile
from typing import Dict, Union
from src.common.robot_state import ROBOT_STATES
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R


def zipped_img_generator(filename, max_samples=1000):
    n_samples = 0
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar:
            if (
                member.isfile() and ".pkl" in member.name
            ):  # Replace 'your_condition' with actual condition
                with tar.extractfile(member) as f:
                    if f is not None:
                        content = f.read()
                        data = pickle.loads(content)
                        n_samples += 1

                        yield data

                        if n_samples >= max_samples:
                            break


def resize(img: Union[np.ndarray, torch.Tensor]):
    """Resizes `img` into 320x240."""
    from torchvision.transforms import InterpolationMode, functional as F

    th, tw = 240, 320
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        img = img.permute(0, 3, 1, 2)

    img = F.resize(
        img, (th, tw), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    if isinstance(img, torch.Tensor):
        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)

    if was_numpy:
        img = img.numpy()

    return img


def resize_crop(img: Union[np.ndarray, torch.Tensor]):
    """
    Resizes `img` and center crops into 320x240.

    Assumes that the channel is last.
    """
    from torchvision.transforms import InterpolationMode, functional as F

    # Must account for maybe having batch dimension
    th, tw = 240, 320
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        img = img.permute(0, 3, 1, 2)
        ch, cw = img.shape[-2:]

    # Calculate the aspect ratio of the original image.
    aspect_ratio = cw / ch

    # Resize based on the width, keeping the aspect ratio constant.
    new_width = int(th * aspect_ratio)
    img = F.resize(
        img, (th, new_width), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    # Calculate the crop size.
    crop_size = max(0, (new_width - tw) // 2)

    if isinstance(img, torch.Tensor):
        img = img[..., crop_size : new_width - crop_size]

        # Move channels back (B, C, H, W) -> (B, H, W, C)
        img = img.permute(0, 2, 3, 1)

    if was_numpy:
        img = img.numpy()

    return img


def clip_quat_xyzw_magnitude(
    delta_quat_xyzw: np.ndarray,
    clip_mag=0.35,
    episode_scale_factor=None,
    per_action=False,
) -> np.ndarray:
    """Clip quaternion rotation magnitudes with explicit legacy semantics.

    ``episode_scale_factor`` is used by timestamp-aligned segments to preserve
    the exact scale computed from their unsplit source episode. Canonical new
    data sets ``per_action=True`` so trajectory length cannot change an
    individual command; legacy callers retain episode-level clipping.
    """
    assert delta_quat_xyzw.shape[-1] == 4
    if clip_mag < 0:
        raise ValueError(f"clip_mag must be nonnegative, got {clip_mag}.")

    delta_rotvec = R.from_quat(delta_quat_xyzw).as_rotvec()
    if episode_scale_factor is not None:
        scale_factor = float(episode_scale_factor)
        if not np.isfinite(scale_factor) or not 0.0 < scale_factor <= 1.0:
            raise ValueError(
                "episode_scale_factor must be finite and in the interval (0, 1]"
            )
    elif per_action:
        magnitude = np.linalg.norm(delta_rotvec, axis=-1, keepdims=True)
        safe_magnitude = np.maximum(magnitude, np.finfo(delta_rotvec.dtype).eps)
        scale_factor = np.minimum(1.0, clip_mag / safe_magnitude)
    else:
        magnitude = np.linalg.norm(delta_rotvec)
        scale_factor = min(1.0, clip_mag / magnitude) if magnitude > 0 else 1.0
    delta_rotvec = scale_factor * delta_rotvec

    delta_quat_xyzw = R.from_rotvec(delta_rotvec).as_quat()

    return delta_quat_xyzw


def filter_and_concat_robot_state(robot_state: Dict[str, torch.Tensor]):
    current_robot_state = []
    for rs in ROBOT_STATES:
        if rs not in robot_state:
            continue

        # if rs == "gripper_width":
        #     robot_state[rs] = robot_state[rs].reshape(-1, 1)
        current_robot_state.append(robot_state[rs])
    return torch.cat(current_robot_state, dim=-1)
