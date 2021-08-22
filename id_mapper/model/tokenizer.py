from math import floor, log, sqrt
from typing import List

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch import nn

from id_mapper.model.common import C2, FeedForward


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: int,
            token_size: int,
            dropout: float = 0.0
    ):
        super(Tokenizer, self).__init__()

        self.image_size = image_size
        self.token_size = token_size

        self.image_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.kernel_size = kernel_size

        deep = floor(log(image_size, 2)) - floor(log(sqrt(self.kernel_size), 2))

        c2s = []
        pre_channel_size = 3
        for i in range(deep):
            out_channel_size = floor(token_size - ((deep - i - 1) * (token_size - 3) / deep))

            c2 = C2(
                ch_in=pre_channel_size,
                ch_out=out_channel_size,
                dropout=dropout
            )
            c2s.append(c2)
            pre_channel_size = out_channel_size

        self.c2 = nn.Sequential(*c2s)

        sample = self.c2(torch.zeros(1, 3, self.image_size, self.image_size))
        batch, kernel, w, h = sample.size()

        self.feature_embeding = nn.Linear(w * h, 1)
        self.feed_forward = FeedForward(
            d_model=kernel,
            intermediate_size=kernel * 2,
            dropout=dropout
        )

        self.__device = torch.device('cpu')

    def to(self, device):
        super(Tokenizer, self).to(device)
        self.__device = device

    def forward(self, images: List[Image]):
        images = self.resizes(images)

        tensor = self.images_to_tensor(images)
        tensor = self.normalizes(tensor)
        tensor = tensor.to(self.__device)

        features = self.c2(tensor)

        batch, kernel, w, h = features.size()

        features = features.view(batch, kernel, w * h)
        features = self.feature_embeding(features)

        features = features.view(batch, kernel)

        tokens = self.feed_forward(features)
        tokens = tokens.view(batch, -1)

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

