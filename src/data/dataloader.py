from random import shuffle
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

from src.common_types import size_2_t
from src.data.dataset import InstanceDataset


class CompareDataLoader(data.Dataset):
    def __init__(
            self,
            dataset: InstanceDataset,
            image_size: size_2_t,
            image_num: int = 2
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.__dataset = dataset
        self.image_size = image_size
        self.image_num = image_num

        self.__image_to_tensor = transforms.ToTensor()
        self.__normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def shuffle(self):
        self.__dataset.shuffle()

    def __len__(self):
        return len(self.__dataset) // self.image_num

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sets = []
        for i in range(self.image_num):
            set, _ = self.__dataset[idx * self.image_num + i]
            sets.append(set)

        for set in sets:
            shuffle(set)

        sets = list(map(lambda it: [image.resize(self.image_size) for image in it], sets))
        sets = list(map(lambda it: self.__images_to_tensor(it), sets))
        sets = list(map(lambda it: self.__normalizes(it), sets))

        keys = torch.cat(sets, dim=0)
        queries = keys

        labels = []
        for i in range(len(sets)):
            current_labels = []
            for j in range(len(sets)):
                if i == j:
                    current_labels.extend([1.0] * len(sets[j]))
                else:
                    current_labels.extend([0.0] * len(sets[j]))

            current_labels = torch.tensor(current_labels)
            current_labels = current_labels.repeat(len(sets[i]), 1)

            labels.append(current_labels)

        labels = torch.cat(labels, dim=0)

        return keys, queries, labels

    def __images_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for image in images:
            tensors.append(self.__image_to_tensor(image))

        return torch.stack(tensors)

    def __normalizes(self, tensor: torch.Tensor) -> torch.Tensor:
        result = []
        for image in tensor:
            result.append(self.__normalize(image))

        return torch.stack(result)
