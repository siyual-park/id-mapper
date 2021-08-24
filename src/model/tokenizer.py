import torch
from torch import nn

from src.model.cbam import CBAM
from src.model.common import Bottleneck, Shortcut, autopad
from src.model.common import Conv
from src.common_types import size_2_t


class ResBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: size_2_t = 3,
            groups: int = 1,
            pooling_kernel_size: size_2_t = 2,
            pooling_stride: size_2_t = 2,
            pooling_dilation: size_2_t = 1,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.conv = Bottleneck(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=groups,
            expansion=expansion,
            dropout_prob=dropout_prob,
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            dilation=pooling_dilation,
            padding=autopad(pooling_kernel_size)
        )

        self.shortcut = Shortcut()

    def forward(self, x):
        x_out = self.conv(x)
        x_out = self.shortcut(x, x_out)
        x_out = self.pooling(x_out)

        return x_out


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: size_2_t,
            token_size: int,
            deep: int = 2,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        channels = token_size // 2

        self.up_scaling = nn.Sequential(
            Conv(
                in_channels=3,
                out_channels=channels,
                kernel_size=1,
                dropout_prob=dropout_prob
            ),
            CBAM(
                gate_channels=channels,
                dropout_prob=dropout_prob
            )
        )

        pooling_kernel_size = 3
        pooling_dilation = 2

        res_blocks = [
            ResBlock(
                channels=channels,
                dropout_prob=dropout_prob,
                pooling_kernel_size=pooling_kernel_size,
                pooling_stride=pooling_kernel_size,
                pooling_dilation=pooling_dilation
            )
            for _ in range(deep)
        ]
        self.compression = nn.Sequential(
            *res_blocks,
            CBAM(
                gate_channels=channels,
                dropout_prob=dropout_prob
            )
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        sample = self.compression(torch.zeros((1, channels, image_size[0], image_size[1])))
        batch, channels, w, h = sample.size()

        self.feature_compression = nn.Linear(w * h, 1)

    def forward(self, x):
        # x (batch, channel, w, h)

        x_out = self.up_scaling(x)
        x_out = self.compression(x_out)

        batch, channel, w, h = x_out.size()
        x_out = x_out.view(batch * channel, -1)

        x_out = self.feature_compression(x_out)
        x_out = x_out.view(batch, channel)

        return x_out




