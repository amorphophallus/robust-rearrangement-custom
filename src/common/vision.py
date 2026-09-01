import torch
import torch.nn as nn
import torchvision

# torchvision.disable_beta_transforms_warning()

from torchvision import transforms
import torchvision.transforms.functional as F

# from torchvision.transforms import v2 as transforms
from ipdb import set_trace as bp  # noqa

class FrontCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode
        self.margin = 20
        self.crop_size = (224, 224)
        self.input_size = (240, 320)
        
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
            x = F.center_crop(x, self.crop_size)
        return x

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"

class WristCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode
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
            
            # 关键：必须分开 Resize 保持深度精度
            rgb = F.resize(rgb, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            depth = F.resize(depth, self.target_size, interpolation=transforms.InterpolationMode.NEAREST)
            return torch.cat([rgb, depth], dim=1)
        else:
            if self.mode == "train":
                x = self.rgb_augment(x)
            return F.resize(x, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"
