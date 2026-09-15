import torch
import torch.nn as nn
import torchvision

# torchvision.disable_beta_transforms_warning()

from torchvision import transforms
import torchvision.transforms.functional as F

# from torchvision.transforms import v2 as transforms
from ipdb import set_trace as bp  # noqa


LEGACY_224_SPATIAL_TRANSFORM = "legacy-224"
CENTER_CROP_224_SPATIAL_TRANSFORM = "center-crop-224"
NO_SPATIAL_TRANSFORM = "none"
SUPPORTED_IMAGE_SPATIAL_TRANSFORMS = (
    LEGACY_224_SPATIAL_TRANSFORM,
    CENTER_CROP_224_SPATIAL_TRANSFORM,
    NO_SPATIAL_TRANSFORM,
)
NATIVE_RGBD_SIZE = (240, 320)


def _validate_spatial_transform(spatial_transform: str) -> None:
    if spatial_transform not in SUPPORTED_IMAGE_SPATIAL_TRANSFORMS:
        raise ValueError(
            "Unsupported image spatial transform "
            f"{spatial_transform!r}; expected one of "
            f"{SUPPORTED_IMAGE_SPATIAL_TRANSFORMS}."
        )


def _validate_native_size(x: torch.Tensor, camera_name: str) -> None:
    spatial_shape = tuple(x.shape[-2:])
    if spatial_shape != NATIVE_RGBD_SIZE:
        raise ValueError(
            f"{camera_name} camera spatial transform 'none' requires native "
            f"240x320 input, got {spatial_shape}."
        )


class FrontCameraTransform(nn.Module):
    def __init__(
        self,
        mode="train",
        spatial_transform=LEGACY_224_SPATIAL_TRANSFORM,
    ):
        super().__init__()
        _validate_spatial_transform(spatial_transform)
        self.mode = mode
        self.spatial_transform = spatial_transform
        self.margin = 20
        self.crop_size = (224, 224)
        self.input_size = NATIVE_RGBD_SIZE

        self.rgb_augment = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W)
        c = x.shape[1]

        if self.mode == "train":
            if c == 4:
                rgb, depth = x[:, :3, ...], x[:, 3:, ...]
                rgb = self.rgb_augment(rgb)
                x = torch.cat([rgb, depth], dim=1)
            else:
                x = self.rgb_augment(x)

            if self.spatial_transform == NO_SPATIAL_TRANSFORM:
                _validate_native_size(x, "Front")
                return x

            # A canonical cross-simulator frame is already 224x224. Applying
            # the legacy 240x280 pre-crop would pad it before random cropping,
            # introducing artificial black borders.
            if tuple(x.shape[-2:]) != self.crop_size:
                if x.shape[-2] < self.crop_size[0] or x.shape[-1] < self.crop_size[1]:
                    raise ValueError(
                        "Front camera input is too small for a 224x224 crop: "
                        f"{tuple(x.shape[-2:])}"
                    )
                if (
                    x.shape[-2] >= self.input_size[0]
                    and x.shape[-1] >= self.input_size[1] - 2 * self.margin
                ):
                    # Preserve the historical augmentation for 240x320 inputs.
                    x = F.center_crop(
                        x,
                        (self.input_size[0], self.input_size[1] - 2 * self.margin),
                    )
                # RandomCrop.get_params uses one crop for the whole batch, so
                # RGB and depth stay geometrically synchronized.
                i, j, h, w = transforms.RandomCrop.get_params(
                    x, output_size=self.crop_size
                )
                x = F.crop(x, i, j, h, w)
        else:
            if self.spatial_transform == NO_SPATIAL_TRANSFORM:
                _validate_native_size(x, "Front")
                return x
            x = F.center_crop(x, self.crop_size)
        return x

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"


class WristCameraTransform(nn.Module):
    def __init__(
        self,
        mode="train",
        spatial_transform=LEGACY_224_SPATIAL_TRANSFORM,
    ):
        super().__init__()
        _validate_spatial_transform(spatial_transform)
        self.mode = mode
        self.spatial_transform = spatial_transform
        self.target_size = (224, 224)

        self.rgb_augment = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]

        if c == 4:
            rgb, depth = x[:, :3, ...], x[:, 3:, ...]
            if self.mode == "train":
                rgb = self.rgb_augment(rgb)

            if self.spatial_transform == NO_SPATIAL_TRANSFORM:
                _validate_native_size(x, "Wrist")
                return torch.cat([rgb, depth], dim=1)

            if self.spatial_transform == CENTER_CROP_224_SPATIAL_TRANSFORM:
                rgb = F.center_crop(rgb, self.target_size)
                depth = F.center_crop(depth, self.target_size)
                return torch.cat([rgb, depth], dim=1)

            # 关键：必须分开 Resize 保持深度精度
            rgb = F.resize(
                rgb,
                self.target_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            depth = F.resize(
                depth,
                self.target_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            return torch.cat([rgb, depth], dim=1)
        else:
            if self.mode == "train":
                x = self.rgb_augment(x)
            if self.spatial_transform == NO_SPATIAL_TRANSFORM:
                _validate_native_size(x, "Wrist")
                return x
            if self.spatial_transform == CENTER_CROP_224_SPATIAL_TRANSFORM:
                return F.center_crop(x, self.target_size)
            return F.resize(
                x,
                self.target_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"
