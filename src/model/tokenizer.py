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

        self.conv1 = Bottleneck(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=groups,
            expansion=expansion,
            dropout_prob=dropout_prob,
        )
        self.conv2 = Bottleneck(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=groups,
            expansion=expansion,
            dropout_prob=dropout_prob,
        )

        self.shortcut = Shortcut()

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)

        return self.shortcut(x, x_out)


class FeatureCompression(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: size_2_t = 3,
            groups: int = 1,
            pooling_kernel_size: size_2_t = 2,
            pooling_stride: size_2_t = 2,
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
            padding=autopad(pooling_kernel_size, None, 1)
        )

    def forward(self, x):
        x_out = self.res_block(x)
        x_out = self.self.attention(x_out)
        x_out = self.self.pooling(x_out)

        return x_out


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: size_2_t,
            token_size: int,
            deep: int = 2
    ):
        super().__init__()

        channels = token_size

        self.up_scaling = nn.Sequential(
            Conv(
                in_channels=3,
                out_channels=channels,
                kernel_size=1
            ),
            CBAM(
                gate_channels=channels
            )
        )

        self.compression = nn.Sequential(*[FeatureCompression(channels=channels) for _ in range(deep)])

        if not isinstance(image_size, int):
            image_size = (image_size, image_size)

    def forward(self, x):
        # x (batch, channel, w, h)

        x_out = self.up_scaling(x)
        x_out = self.compression(x_out)

        batch, channel, w, h = x_out.size()
        x_out = x_out.view(batch, channel, -1)

        return x_out




