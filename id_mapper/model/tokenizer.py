from math import floor, log
from typing import List

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch import nn


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
            act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, autopad(kernel, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: int,
            token_size: int
    ):
        super(Tokenizer, self).__init__()

        deep = floor(log(image_size, 2))

        self.image_size = image_size
        self.token_size = token_size

        self.image_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        convs = []
        pre_channel_size = 3
        for i in range(deep):
            out_channel_size = floor(token_size - ((deep - i - 1) * (token_size - 3) / deep))

            conv = nn.Sequential(
                Conv(ch_in=pre_channel_size, ch_out=out_channel_size, padding="same"),
                Conv(ch_in=out_channel_size, ch_out=out_channel_size, padding="same"),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            convs.append(conv)
            pre_channel_size = out_channel_size

        self.conv = nn.Sequential(*convs)

        sample = self.conv(torch.zeros(1, 3, self.image_size, self.image_size))
        batch, kernel, w, h = sample.size()

        self.conv_output_size = w * h

        self.linear = nn.Linear(self.conv_output_size, 1)

    def forward(self, images: List[Image]):
        images = self.resizes(images)

        tensor = self.images_to_tensor(images)
        tensor = self.normalizes(tensor)

        features = self.conv(tensor)

        tokens = self.mapping(features)
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
