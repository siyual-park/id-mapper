from typing import Optional

import numpy as np
import torch
from torch import nn

from src.common_types import size_2_t


def autopad(kernel_size: size_2_t, padding: Optional[size_2_t], dilation: size_2_t) -> size_2_t:
    # Pad to 'same'
    if padding is None:
        kernel_size = np.asarray((kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size)
        dilation = np.asarray((dilation, dilation) if isinstance(dilation, int) else dilation)

        padding = (dilation * (kernel_size - 1) + 1) // 2

    return padding


class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: size_2_t,
            stride: size_2_t = 1,
            padding: Optional[size_2_t] = None,
            dilation: size_2_t = 1,
            groups: int = 1,
            activate: bool or nn.Module = True,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU() if activate is True else (
            activate if isinstance(activate, nn.Module) else nn.Identity()
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        return self.dropout(self.activate(self.batch_norm(self.conv(x))))

    def forward_fuse(self, x):
        return self.activate(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: size_2_t = 3,
            groups: int = 1,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        down_sample_channels = int(out_channels * expansion)

        self.down_sample = Conv(
            in_channels,
            down_sample_channels,
            kernel_size=1
        )
        self.conv = Conv(
            down_sample_channels,
            down_sample_channels,
            kernel_size=kernel_size,
            groups=groups,
            dropout_prob=dropout_prob
        )
        self.up_sample = Conv(
            down_sample_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.up_sample(self.conv(self.down_sample(x)))


class Shortcut(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x.size() != y.size():
            return y

        return x + y


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CosineSimilarly(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1_norm = x1 / x1.norm(dim=1)[:, None]
        x2_norm = x2 / x2.norm(dim=1)[:, None]

        return torch.mm(x1_norm, x2_norm.transpose(0, 1))


class CosineDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarly = CosineSimilarly()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        cosine_similarly = self.cosine_similarly(x1, x2)

        return (1 - cosine_similarly) / 2
