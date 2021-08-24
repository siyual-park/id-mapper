from pathlib import Path
from random import shuffle
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools import coco
from torch.utils import data

from src.common_types import size_2_t
from src.data.utils import get_data_size, represents_int


def load_annotated_ids(coco: coco.COCO):
    whole_image_ids = coco.getImgIds()

    image_ids = []

    for id in whole_image_ids:
        annotations_ids = coco.getAnnIds(imgIds=id, iscrowd=False)
        if len(annotations_ids) > 0:
            image_ids.append(id)

    return image_ids


class COCODataset(data.Dataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str
    ):
        path = Path(path)

        self.dataset = dataset
        self.annotations_path = path.joinpath('annotations')
        self.data_path = path.joinpath(dataset)

        self.__coco = coco.COCO(
            self.annotations_path
                .joinpath(f'instances_{dataset}.json')
        )

        self.__image_ids = load_annotated_ids(self.__coco)

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[Image.Image, np.ndarray]:
        id = self.__image_ids[idx]

        image = self.load_image(id)
        annotations = self.load_annotations(id)

        return image, annotations

    def load_image(self, image_id) -> Image.Image:
        image_info = self.__coco.loadImgs(image_id)[0]
        path = self.data_path.joinpath(image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image

    def load_annotations(self, image_id) -> np.ndarray:
        # get ground truth annotations
        annotations_ids = self.__coco.getAnnIds(imgIds=image_id, iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.__coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id']
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class InstanceDataset(data.Dataset):
    def __init__(
            self,
            path: str or Path,
            dataset: str
    ):
        path = Path(path)

        self.dataset = dataset
        self.data_path = path.joinpath(dataset)

        data_size = get_data_size(self.data_path)
        self.__image_ids = list(range(data_size))

    def shuffle(self):
        shuffle(self.__image_ids)

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx) -> Tuple[List[Image.Image], Path]:
        id = self.__image_ids[idx]

        images_path = self.data_path.joinpath(str(id))

        images = []
        for entry in sorted(images_path.iterdir()):
            if represents_int(entry.name.removesuffix(entry.suffix)):
                image = Image.open(entry).convert('RGB')
                images.append(image)

        return images, images_path


class CompareDataset(data.Dataset):
    def __init__(
            self,
            dataset: InstanceDataset,
            image_size: size_2_t,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.__dataset = dataset
        self.image_size = image_size

        self.__image_to_tensor = transforms.ToTensor()
        self.__normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def shuffle(self):
        self.__dataset.shuffle()

    def __len__(self):
        return len(self.__dataset) // 2

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        set1, _ = self.__dataset[idx * 2]
        set2, _ = self.__dataset[idx * 2 + 1]

        shuffle(set1)
        shuffle(set2)

        set1 = [image.resize(self.image_size) for image in set1]
        set2 = [image.resize(self.image_size) for image in set2]

        set1 = self.__images_to_tensor(set1)
        set2 = self.__images_to_tensor(set2)

        set1 = self.__normalizes(set1)
        set2 = self.__normalizes(set2)

        keys = torch.cat([set1, set2], dim=0)
        queries = keys

        set1_labels = [1] * len(set1)
        set1_labels.extend([0] * len(set2))
        set1_labels = torch.tensor(set1_labels)

        set2_labels = 1 - set1_labels

        set1_labels = set1_labels.repeat(len(set1), 1)
        set2_labels = set2_labels.repeat(len(set2), 1)

        labels = torch.cat([set1_labels, set2_labels], dim=1)

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
