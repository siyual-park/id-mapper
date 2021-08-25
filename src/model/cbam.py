import torch
from torch import nn
from torch.nn import functional as F

from src.model.common import CalculateChannel, Conv


class ChannelPool(nn.Module):
    def forward(self, x):
        # x (batch, channel, w, h)
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)

        return torch.cat((max_pool, avg_pool), dim=1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelAttention(nn.Module):
    def __init__(
            self,
            gate_channels: int,
            reduction_ratio: float = 1 / 16,
            pool_types=None,
            dropout_prob: float = 0.0
    ):
        super(ChannelAttention, self).__init__()

        if pool_types is None:
            pool_types = ['avg', 'max']

        self.gate_channels = gate_channels

        middle_channels = max(int(gate_channels * reduction_ratio), 2)
        self.mlp = CalculateChannel(nn.Sequential(
            nn.Linear(gate_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, gate_channels),
            nn.Dropout(dropout_prob)
        ))

        self.pool_types = pool_types

    def forward(self, x):
        # x (batch, channel, w, h)

        channel_att_sum = None

        for pool_type in self.pool_types:
            channel_att_raw = None

            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_raw is not None:
                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

        channel_att_sum = F.sigmoid(channel_att_sum)
        channel_att_sum = channel_att_sum.expand_as(x)

        return channel_att_sum


class SpatialAttention(nn.Module):
    def __init__(self, dropout_prob: float = 0.0):
        super(SpatialAttention, self).__init__()

        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            activate=False,
            dropout_prob=dropout_prob
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        return F.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(
            self,
            gate_channels: int,
            reduction_ratio: float = 1 / 16,
            pool_types=None,
            no_spatial: bool = False,
            dropout_prob: float = 0.0
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            gate_channels,
            reduction_ratio,
            pool_types,
            dropout_prob=dropout_prob
        )
        self.spatial_attention = SpatialAttention(
            dropout_prob=dropout_prob
        ) if not no_spatial else None

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x_out = x * channel_attention

        if self.spatial_attention is not None:
            spatial_attention = self.spatial_attention(x_out)
            x_out = x_out * spatial_attention

        return x_out
