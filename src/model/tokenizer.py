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

        self.shortcut = Shortcut()

    def forward(self, x):
        x_out = self.conv(x)

        return self.shortcut(x, x_out)


class FeatureCompression(nn.Module):
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

        self.res_block = ResBlock(
            channels,
            kernel_size,
            groups,
            expansion,
            dropout_prob
        )
        self.attention = CBAM(
            gate_channels=channels
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            dilation=pooling_dilation,
            padding=autopad(pooling_kernel_size)
        )

    def forward(self, x):
        x_out = self.res_block(x)
        x_out = self.attention(x_out)
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

        self.compression = nn.Sequential(*[
            FeatureCompression(
                channels=channels,
                dropout_prob=dropout_prob,
                pooling_kernel_size=pooling_kernel_size,
                pooling_stride=pooling_kernel_size,
                pooling_dilation=pooling_dilation
            )
            for _ in range(deep)
        ])

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        sample = self.compression(torch.zeros((1, channels, image_size[0], image_size[1])))
        batch, channels, w, h = sample.size()

        self.feature_compression = nn.Sequential(
            nn.Linear(w * h, int(w * h * 0.5)),
            nn.ReLU(),
            nn.Linear(int(w * h * 0.5), 1),
        )

        self.feature_expand = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, token_size),
        )

    def forward(self, x):
        # x (batch, channel, w, h)

        x_out = self.up_scaling(x)
        x_out = self.compression(x_out)

        batch, channel, w, h = x_out.size()
        x_out = x_out.view(batch * channel, -1)

        x_out = self.feature_compression(x_out)

        x_out = x_out.view(batch, channel)
        x_out = self.feature_expand(x_out)

        return x_out




