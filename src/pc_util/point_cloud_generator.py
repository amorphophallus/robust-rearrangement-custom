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
        bbox_half_extent = 0.2,
        debug: bool = False,
    ):
        self.env = env
        self.camera_name = camera_name
        self.target_frame = target_frame
        self.max_points = max_points
        self.debug = debug
        self.device = getattr(env, "device", torch.device("cpu"))
        # Optional axis-aligned 3D bbox cropping centered at EE pose
        # If set to a scalar, uses the same half-extent for x/y/z; if an iterable of len 3, uses per-axis.
        self.bbox_half_extent = bbox_half_extent
        # Tracks which EE pose source was used last (for debug)
        self._last_eepose_source: Optional[str] = None

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
        """Build camera-to-world transform using a stable right-handed look-at basis.

        - forward (+Z camera) = cam_target - cam_pos
        - right = cross(forward, up_ref)
        - up    = cross(right, forward)
        Uses env.front_cam_up if provided; falls back to +Y, then +Z if colinear.
        """
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

        up_ref = getattr(self.env, "front_cam_up", None)
        if up_ref is None:
            up_ref = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        else:
            up_ref = torch.tensor(up_ref, device=self.device, dtype=torch.float32)

        right = torch.cross(forward, up_ref)
        if right.norm() < 1e-6:
            up_ref = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            right = torch.cross(forward, up_ref)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward)

        R = torch.stack([right, up, forward], dim=1)
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        return T

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
        """Unproject depth pixels back to world coordinates using the exact P, V pipeline.

        This is the algebraic inverse of the provided projection:
        clip = P @ (V @ pw)
        ndc = clip / clip[3]
        x_img = (1 - (ndc[1] * 0.5 + 0.5)) * W
        y_img = (ndc[0] * 0.5 + 0.5) * H

        Depth entering this function has already been sign-flipped; we explicitly take
        another negation so camera-forward depth is positive and view-space Z is negative.
        """
        H, W = depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )

        # Depth was pre-flipped; enforce camera-forward positive depth here.
        z_cam = -depth[mask].float()
        xs = xs[mask].float()
        ys = ys[mask].float()

        # Recover NDC from pixel coordinates using the same projection mapping.
        ndc0 = 2.0 * (ys / float(H)) - 1.0
        ndc1 = 1.0 - 2.0 * (xs / float(W))

        # Assemble clip-space points on the ray (z set to +1; we scale later with depth).
        ones = torch.ones_like(z_cam)
        clip = torch.stack([ndc0, ndc1, ones, ones], dim=-1)

        # Inverse projection and view to get a ray direction in view space.
        P, V = self.env.get_front_projection_view_matrix()
        P_t = torch.tensor(P, device=self.device, dtype=torch.float32).reshape(4, 4)
        V_t = torch.tensor(V, device=self.device, dtype=torch.float32).reshape(4, 4)
        invP = torch.inverse(P_t)
        invV = torch.inverse(V_t)

        view_h = (invP @ clip.t()).t()
        view = view_h[:, :3] / view_h[:, 3:4]

        # Scale the view-space ray so its Z matches the desired camera-forward depth.
        scale = (-z_cam) / (view[:, 2] + 1e-8)
        view_scaled = view * scale.unsqueeze(1)

        view_scaled_h = torch.cat([view_scaled, ones.unsqueeze(1)], dim=-1)
        world_h = (invV @ view_scaled_h.t()).t()
        pts_world = world_h[:, :3] / world_h[:, 3:4]
        return pts_world
    
    def _transform(self, pts_world: torch.Tensor, target_frame: Optional[str] = None):
        """Transform world-frame points to the requested target frame.

        Alternative implementation using direct robot base position subtraction.
        target_frame: "world" or "robot". Defaults to `self.target_frame`.
        假设 robot base 没有旋转，只是平移，和 Furniture Bench 中的设定一致。
        """
        if target_frame is None:
            target_frame = self.target_frame
        if target_frame == "robot":
            # Get robot base position from rb_states directly
            if not hasattr(self.env, "rb_states") or not hasattr(self.env, "base_idxs"):
                raise RuntimeError("Env must have rb_states and base_idxs for robot frame transform.")
            # rb_states: (num_rigid_bodies, 13) in world space, [:3] is position
            base_pos = self.env.rb_states[self.env.base_idxs, :3][0].detach().to(self.device)  # Take first env
            # Transform to robot frame: subtract base position (assume no rotation)
            pts_robot = pts_world - base_pos
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

    def _get_world_eepose_center(self) -> torch.Tensor:
        """Get EE pose center (position) in world (sim) frame directly from rb_states.

        rb_states contains rigid body states in global/sim space.
        Returns the EE position as a 3D tensor in world frame.
        """
        if not hasattr(self.env, "rb_states") or not hasattr(self.env, "ee_idxs"):
            raise RuntimeError("Env must have rb_states and ee_idxs for world EE pose.")
        # rb_states: (num_rigid_bodies, 13) in SIM/global space
        # [:3] is position, [3:7] is quaternion
        hand_pos = self.env.rb_states[self.env.ee_idxs, :3]
        # hand_pos shape: (num_envs, 3), take first env
        pos = hand_pos[0].detach().to(self.device)
        self._last_eepose_source = "rb_states:world"
        return pos

    def _apply_3d_bbox_crop(self, depth: torch.Tensor, half_extent) -> torch.Tensor:
        """Compute a boolean mask of pixels whose 3D points lie inside an axis-aligned bbox.

        The bbox is centered at the EE pose. This function computes its own validity mask
        based on positive, finite depth and does NOT require the caller's mask.
        """
        if half_extent is None:
            # If no bbox, keep everything (let caller intersect as needed)
            return torch.ones_like(depth, dtype=torch.bool)

        base_mask = torch.isfinite(depth) & (depth > 0)
        K = self._intrinsics()
        # _unproject now returns world-frame points directly
        pts_world = self._unproject(depth, K, base_mask)

        # Half extents
        if isinstance(half_extent, (int, float)):
            hx, hy, hz = float(half_extent), float(half_extent), float(half_extent)
        else:
            he = list(half_extent)
            if len(he) != 3:
                raise ValueError("bbox_half_extent must be a scalar or a length-3 iterable")
            hx, hy, hz = float(he[0]), float(he[1]), float(he[2])

        center_world = self._get_world_eepose_center()  # world (sim) frame directly
        
        lower = center_world - torch.tensor([hx, hy, hz], device=self.device, dtype=torch.float32)
        upper = center_world + torch.tensor([hx, hy, hz], device=self.device, dtype=torch.float32)

        keep = (pts_world >= lower).all(dim=-1) & (pts_world <= upper).all(dim=-1)

        # Map kept points back to full-resolution mask
        ys, xs = torch.where(base_mask)
        bbox_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        if keep.numel() > 0:
            bbox_mask[ys[keep], xs[keep]] = True
        # Save for visualization
        self._last_bbox_mask = bbox_mask
        return bbox_mask

    def _visualize_bbox_crop_on_image(self, env_idx: int, bbox_mask: torch.Tensor):
        """Fetch color image and show it with cropped-out pixels colored black.

        Minimal logic: get `IMAGE_COLOR`, apply mask, display via OpenCV.
        """
        import numpy as np
        handles = self.env.camera_handles.get(self.camera_name, None)
        if handles is None or len(handles) <= env_idx:
            print("[PointCloudGenerator] No camera handle for color visualization.")
            return
        handle = handles[env_idx]
        # Access color image
        self.env.isaac_gym.start_access_image_tensors(self.env.sim)
        color = gymtorch.wrap_tensor(
            self.env.isaac_gym.get_camera_image_gpu_tensor(
                self.env.sim, self.env.envs[env_idx], handle, gymapi.IMAGE_COLOR
            )
        )
        self.env.isaac_gym.end_access_image_tensors(self.env.sim)

        img = color.detach().cpu()
        # Expect HxWxC; reduce to RGB if RGBA
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
        elif img.ndim == 3 and img.shape[0] >= 3:
            img = img[:3, ...].permute(1, 2, 0)
        # Convert to uint8 if needed (assume [0,1] or [0,255])
        if img.dtype != torch.uint8:
            maxv = float(img.max().item())
            if maxv <= 1.0 + 1e-6:
                img = (img * 255.0).clamp(0, 255)
            img = img.to(torch.uint8)
        img_np = img.numpy().copy()

        bm = bbox_mask.detach().cpu().numpy()
        img_np[~bm] = 0

        try:
            import cv2
            win = f"BBox Crop - {self.camera_name}"
            cv2.imshow(win, img_np[..., ::-1])
            cv2.waitKey(1)
        except Exception:
            print("[PointCloudGenerator] OpenCV not available for display.")

    def _crop(self, depth, seg):
        # Depth mask: valid >0 and finite
        valid = torch.isfinite(depth) & (depth > 0)
        valid_depth_count = int(valid.sum().item())

        # 主逻辑
        if seg is not None:
            allowed = (seg >= 5) | (seg == 4)
            valid = valid & allowed
        # Optional 3D bbox crop centered at EE pose (intersect outside)
        if self.bbox_half_extent is not None:
            bbox_mask = self._apply_3d_bbox_crop(depth, self.bbox_half_extent)
            valid = valid & bbox_mask
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
                f"[PointCloudGenerator] camera={self.camera_name}"
                f" depth_valid={valid_depth_count} after_seg={valid_after_seg}"
                f" depth_shape={tuple(depth.shape)} depth_minmax={depth_stats[:2]}"
                f" depth_finite_frac={depth_stats[2]:.3f} seg_stats={seg_stats}"
            )
            raise ValueError("No Valid Point")
        return valid

    def generate_transformed_cropped_point_cloud(
        self,
        env_idx: int = 0,
        max_points: Optional[int] = None,
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

        K = self._intrinsics()
        mask = self._crop(depth, seg)
        pts_world = self._unproject(depth, K, mask)
        pts_frame = self._transform(pts_world)
        pts_frame = self._downsample(pts_frame, max_points, mode=downsample_mode)
        if visualize:
            self._visualize_points(pts_frame)
        # debug 展示 crop 完的图片
        if self.debug and self._last_bbox_mask is not None:
            self._visualize_bbox_crop_on_image(env_idx, self._last_bbox_mask)
        return pts_frame

    def generate_transformed_cropped_point_cloud_for_all_env(
        self,
        max_points: Optional[int] = None,
        downsample_mode: str = "fps",
    ) -> List[torch.Tensor]:
        """Generate point clouds for all environments.

        Returns:
            List of (N, 3) tensors, one per environment.
        """
        num_envs = self.env.num_envs
        point_clouds = []
        for env_idx in range(num_envs):
            try:
                pts = self.generate_transformed_cropped_point_cloud(
                    env_idx=env_idx,
                    max_points=max_points,
                    downsample_mode=downsample_mode,
                    visualize=False,
                )
                point_clouds.append(pts)
            except Exception as e:
                print(f"[PointCloudGenerator] Failed to generate PC for env {env_idx}: {e}")
                point_clouds.append(torch.empty((0, 3), device=self.device))
        return point_clouds
