from math import floor, log
from typing import List

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch import nn

from id_mapper.model.attention import SelfAttention


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: int,
            token_size: int,
            head_size: int,
            intermediate_size: int,
            attention_size: int,
            dropout: float = 0.0
    ):
        super(Tokenizer, self).__init__()

        kernel_size = token_size // head_size

        self.image_size = image_size
        self.token_size = token_size

        self.image_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        deep = floor(log(image_size, 2))

        embedding = []
        pre_channel_size = 3
        for i in range(deep):
            out_channel_size = floor(token_size - ((deep - i - 1) * (token_size - 3) / deep))

            c2 = C2(
                ch_in=pre_channel_size,
                ch_out=out_channel_size,
                dropout=dropout
            )
            embedding.append(c2)
            pre_channel_size = out_channel_size

        self.embedding = nn.Sequential(*embedding)

        sample = self.embedding(torch.zeros(1, 3, self.image_size, self.image_size))
        batch, kernel, w, h = sample.size()

        self.embedding_output_size = w * h

        self.linear = nn.Linear(self.embedding_output_size, 1)

        attentions = []
        for i in range(attention_size):
            attentions.append(SelfAttention(
                kernel_size=kernel_size,
                head_size=head_size,
                dropout=dropout,
                intermediate_size=intermediate_size
            ))

        self.attentions = nn.Sequential(*attentions)

        self.__device = torch.device('cpu')

    def to(self, device):
        self.__device = device
        super(Tokenizer, self).to(device)

    def forward(self, images: List[Image]):
        images = self.resizes(images)

        tensor = self.images_to_tensor(images)
        tensor = self.normalizes(tensor)
        tensor = tensor.to(self.__device)

        features = self.embedding(tensor)

        tokens = self.mapping(features)
        tokens = self.attentions(tokens)
        tokens = torch.sigmoid(tokens)

        return tokens

    def resizes(self, images: List[Image]) -> List[Image]:
        result = []
        for image in images:
            result.append(image.resize((self.image_size, self.image_size)))

        return result

    def images_to_tensor(self, images: List[Image]) -> torch.Tensor:
        tensors = []
        for image in images:
            tensors.append(self.image_to_tensor(image))

        return torch.stack(tensors)

    def normalizes(self, tensor: torch.Tensor) -> torch.Tensor:
        result = []
        for image in tensor:
            result.append(self.normalize(image))

        return torch.stack(result)

    def mapping(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, kernel, w, h = tensor.size()

        tensor = tensor.view(-1, w * h)
        output = self.linear(tensor)
        return output.view(batch, kernel)
