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


class Upscaling(nn.Module):
    def __init__(
            self,
            channels: int,
            pooling_kernel_size: size_2_t = 2,
            pooling_stride: size_2_t = 2,
            pooling_dilation: size_2_t = 1,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.conv = Conv(
            in_channels=3,
            out_channels=channels,
            kernel_size=1,
            dropout_prob=dropout_prob
        )
        self.attention = CBAM(
            gate_channels=channels,
            dropout_prob=dropout_prob
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            dilation=pooling_dilation,
            padding=autopad(pooling_kernel_size)
        )

    def forward(self, x):
        x_out = self.conv(x)
        x_out = self.attention(x_out)
        x_out = self.pooling(x_out)

        return x_out


class Compression(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: size_2_t = 3,
            groups: int = 1,
            pooling_kernel_size: size_2_t = 2,
            pooling_stride: size_2_t = 2,
            pooling_dilation: size_2_t = 1,
            deep: int = 2,
            expansion: float = 0.5,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.res_block = nn.Sequential(*[
            ResBlock(
                channels=channels,
                kernel_size=kernel_size,
                groups=groups,
                pooling_kernel_size=pooling_kernel_size,
                pooling_stride=pooling_kernel_size,
                pooling_dilation=pooling_dilation,
                expansion=expansion,
                dropout_prob=dropout_prob,
            ) for _ in range(deep)
        ])

        self.attention = CBAM(
            gate_channels=channels,
            dropout_prob=dropout_prob
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

        pooling_kernel_size = 3
        pooling_dilation = 2
        pooling_stride = 2

        self.up_scaling = Upscaling(
            channels=channels,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_dilation=pooling_dilation,
            dropout_prob=dropout_prob,
        )

        self.compression = Compression(
            channels=channels,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_dilation=pooling_dilation,
            deep=deep,
            dropout_prob=dropout_prob,
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        zero_image = torch.zeros((1, 3, image_size[0], image_size[1]))

        tmp = self.up_scaling(zero_image)
        tmp = self.compression(tmp)

        batch, channels, w, h = tmp.size()

        self.feature_compression = nn.Linear(w * h, 1)
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
