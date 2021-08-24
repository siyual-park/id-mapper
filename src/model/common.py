from typing import Optional

from torch import nn

from src.model.common_types import size_2_t


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: size_2_t,
            stride: size_2_t = 1,
            padding: Optional[str] = None,
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
            autopad(kernel_size, padding),
            groups=groups,
            bias=False
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.SiLU() if activate is True else (
            activate if isinstance(activate, nn.Module) else nn.Identity())
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
    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module

    def forward(self, x):
        y = self.module(x)

        if x.size() != y.size():
            return y

        return x + y
