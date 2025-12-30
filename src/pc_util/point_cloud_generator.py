import math
from typing import Iterable, List, Optional

import torch
from isaacgym import gymapi, gymtorch

# Optional visualization via Open3D
try:
    import open3d as o3d
except Exception:
    o3d = None

# Optional PyTorch3D FPS
try:
    from pytorch3d.ops import sample_farthest_points as _torch3d_sample_fps
except Exception:
    _torch3d_sample_fps = None


class PointCloudGenerator:
    """Generate point clouds from an Isaac Gym env camera and map to robot base frame."""

    def __init__(
        self,
        env,
        camera_name: str = "front",
        target_frame: str = "robot",
        max_points: int = 4096,
    ):
        self.env = env
        self.camera_name = camera_name
        self.target_frame = target_frame
        self.max_points = max_points
        self.device = getattr(env, "device", torch.device("cpu"))

        # Open3D visualizer state
        self._viz = None
        self._pcd = None
        self._warned_no_o3d = False

        # Ensure camera tensors are available for depth/segmentation
        if not hasattr(self.env, "camera_handles"):
            raise RuntimeError("Env must expose camera_handles; call set_camera first.")

    def _intrinsics(self):
        width, height = self.env.img_size
        fov = getattr(self.env, "camera_cfg", None)
        fov = fov.horizontal_fov if fov is not None else 69.4
        fov_rad = math.radians(float(fov))
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        return K

    def _extrinsics_cam_to_world(self):
        if not hasattr(self.env, "front_cam_pos") or not hasattr(
            self.env, "front_cam_target"
        ):
            raise RuntimeError("Front camera pose not cached on env.")
        cam_pos = torch.tensor(self.env.front_cam_pos, device=self.device, dtype=torch.float32)
        cam_target = torch.tensor(
            self.env.front_cam_target, device=self.device, dtype=torch.float32
        )
        forward = cam_target - cam_pos
        forward = forward / (forward.norm() + 1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = torch.cross(up, forward)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(forward, right)
        R = torch.stack([right, up, forward], dim=1)
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        return T

    def _sim_to_robot(self):
        if not hasattr(self.env, "sim_to_robot_mat"):
            raise RuntimeError("Env missing sim_to_robot_mat for frame transform.")
        return self.env.sim_to_robot_mat.to(self.device)

    def _fetch_depth_and_seg(self, env_idx: int):
        handles = self.env.camera_handles.get(self.camera_name, None)
        if handles is None or len(handles) <= env_idx:
            raise RuntimeError(f"Camera handles missing for {self.camera_name}")
        handle = handles[env_idx]
        depth_tensor = gymtorch.wrap_tensor(
            self.env.isaac_gym.get_camera_image_gpu_tensor(
                self.env.sim, self.env.envs[env_idx], handle, gymapi.IMAGE_DEPTH
            )
        )
        # Segmentation may be unavailable; try and fall back gracefully
        seg_tensor = None
        try:
            seg_tensor = gymtorch.wrap_tensor(
                self.env.isaac_gym.get_camera_image_gpu_tensor(
                    self.env.sim,
                    self.env.envs[env_idx],
                    handle,
                    gymapi.IMAGE_SEGMENTATION,
                )
            )
        except Exception:
            seg_tensor = None
        return depth_tensor, seg_tensor

    def _unproject(self, depth: torch.Tensor, K: torch.Tensor, mask: torch.Tensor):
        H, W = depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        z = depth[mask]
        xs = xs[mask]
        ys = ys[mask]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        pts_cam = torch.stack([x, y, z], dim=-1)
        return pts_cam

    def _transform(self, pts_cam: torch.Tensor):
        T_c2w = self._extrinsics_cam_to_world()
        pts_h = torch.cat([pts_cam, torch.ones_like(pts_cam[..., :1])], dim=-1)
        pts_world = (T_c2w @ pts_h.t()).t()[..., :3]
        if self.target_frame == "robot":
            T_s2r = self._sim_to_robot()
            pts_h = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
            pts_robot = (T_s2r @ pts_h.t()).t()[..., :3]
            return pts_robot
        return pts_world

    def _downsample(self, pts: torch.Tensor, max_points: Optional[int], mode: str = "random"):
        """Downsample points to at most max_points.

        mode:
        - random: random permutation sampling
        - uniform: evenly spaced indices across the array order
        - fps: farthest point sampling based on Euclidean distance
        """
        if max_points is None:
            max_points = self.max_points
        n = pts.shape[0]
        if n <= max_points:
            return pts
        if mode == "uniform":
            # Evenly spaced indices from [0, n-1]
            idx = torch.linspace(0, n - 1, steps=max_points, device=pts.device)
            idx = idx.round().to(torch.long)
        elif mode == "fps":
            # Use PyTorch3D farthest point sampling exclusively
            if _torch3d_sample_fps is None:
                raise RuntimeError("PyTorch3D not available: install pytorch3d to use downsample_mode='fps'.")
            pb = pts[None, :, :].to(torch.float32)
            K_t = torch.as_tensor([max_points], device=pb.device)
            _, sampled_indices = _torch3d_sample_fps(points=pb[..., :3], K=K_t)
            idx = sampled_indices.squeeze(0).to(torch.long)
        else:
            idx = torch.randperm(n, device=pts.device)[:max_points]
        return pts[idx]

    def _visualize_points(self, pts: torch.Tensor):
        """Show/update interactive point cloud window with draggable view."""
        if o3d is None:
            if not self._warned_no_o3d:
                print("[PointCloudGenerator] Open3D not available; visualization disabled.")
                self._warned_no_o3d = True
            return

        pts_np = pts.detach().cpu().numpy()

        if self._viz is None:
            self._viz = o3d.visualization.Visualizer()
            self._viz.create_window(window_name="PointCloud Viewer", width=960, height=720)
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(pts_np)
            self._viz.add_geometry(self._pcd)
            render_opt = self._viz.get_render_option()
            render_opt.point_size = 2.0
        else:
            self._pcd.points = o3d.utility.Vector3dVector(pts_np)
            self._viz.update_geometry(self._pcd)

        self._viz.poll_events()
        self._viz.update_renderer()

    def generate_transformed_cropped_point_cloud(
        self,
        env_idx: int = 0,
        max_points: Optional[int] = None,
        mask_actor_ids: Optional[Iterable[int]] = None,
        downsample_mode: str = "random",
        visualize: bool = False,
    ) -> torch.Tensor:
        # Render and access tensors
        self.env.isaac_gym.render_all_camera_sensors(self.env.sim)
        self.env.isaac_gym.start_access_image_tensors(self.env.sim)
        depth, seg = self._fetch_depth_and_seg(env_idx)
        depth = depth.clone()
        seg = seg.clone() if seg is not None else None
        self.env.isaac_gym.end_access_image_tensors(self.env.sim)

        # isaac gym 会沿着 -z 轴给深度
        depth_flipped = False
        if float(depth.max().item()) <= 0.0:
            depth = -depth
            depth_flipped = True

        # Depth mask: valid >0 and finite
        valid = torch.isfinite(depth) & (depth > 0)
        valid_depth_count = int(valid.sum().item())

        if seg is not None:
            allowed = (seg >= 5) | (seg == 4)
            valid = valid & allowed
        valid_after_seg = int(valid.sum().item())

        # debug output
        if valid_depth_count == 0 or valid_after_seg == 0:
            depth_stats = (
                float(depth.min().item()),
                float(depth.max().item()),
                float(torch.isfinite(depth).float().mean().item()),
            )
            seg_stats = None
            if seg is not None:
                unique_seg = torch.unique(seg).cpu().tolist()
                seg_stats = {
                    "unique_ids": unique_seg[:10],
                    "num_unique": len(unique_seg),
                }
            print(
                f"[PointCloudGenerator] env={env_idx} camera={self.camera_name}"
                f" depth_valid={valid_depth_count} after_seg={valid_after_seg}"
                f" depth_shape={tuple(depth.shape)} depth_minmax={depth_stats[:2]}"
                f" depth_finite_frac={depth_stats[2]:.3f} seg_stats={seg_stats}"
                f" depth_flipped={depth_flipped}"
            )
            return torch.empty((0, 3), device=self.device)

        K = self._intrinsics()
        pts_cam = self._unproject(depth, K, valid)
        pts_frame = self._transform(pts_cam)
        pts_frame = self._downsample(pts_frame, max_points, mode=downsample_mode)
        if visualize:
            self._visualize_points(pts_frame)
        return pts_frame
