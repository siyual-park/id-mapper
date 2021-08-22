import torch
from torch import nn

class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self,
            ch_in,
            ch_out,
            kernel=1,
            stride=1,
            padding=None,
            groups=1,
            act=True,
            dropout=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, autopad(kernel, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.dropout(self.act(self.conv(x)))


class C2(nn.Module):
    # Standard convolution
    def __init__(
            self,
            ch_in,
            ch_out,
            pooling_kernel_size=2,
            pooling_stride=2,
            dropout=0.0
    ):
        super().__init__()

        ch_mid = ch_out // 2

        self.conv1 = Conv(ch_in=ch_in, ch_out=ch_mid, padding="same", dropout=dropout)
        self.conv2 = Conv(ch_in=ch_mid, ch_out=ch_out - ch_mid, padding="same", dropout=dropout)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)

    def forward(self, x):
        feat_1 = self.conv1(x)
        feat_2 = self.conv2(feat_1)

        return self.pooling(torch.cat([feat_1, feat_2], dim=1))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float, intermediate_size: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tensor = self.linear1(inputs)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)

        tensor = self.linear2(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)

        tensor = self.norm(tensor)

        return tensor