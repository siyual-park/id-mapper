import os
from pathlib import Path
from random import shuffle, random, randint

import torch
from PIL import ImageFilter
from PIL.Image import Image
from tqdm import tqdm

from id_mapper.dataset.instance import InstanceImage


def _chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def _random_transform(image: Image, rate: float) -> Image:
    current_rate = random()
    if current_rate <= rate:
        image = __random_rotate(image)

    current_rate = random()
    if current_rate <= rate:
        image = __random_crop(image)

    current_rate = random()
    if current_rate <= rate:
        image = __noise(image)

    current_rate = random()
    if current_rate <= rate:
        image = __random_resize(image)

    return image


def __random_rotate(image: Image) -> Image:
    angle = randint(-5, 5)
    return image.rotate(angle)


def __random_crop(image: Image) -> Image:
    (w, h) = image.size

    x1 = randint(0, w // 10)
    y1 = randint(0, h // 10)
    x2 = randint(w - w // 10, w)
    y2 = randint(h - h // 10, h)

    image = image.crop((x1, y1, x2, y2))
    image = image.resize((w, h))

    return image


def __random_resize(image: Image) -> Image:
    (w, h) = image.size

    w = randint(w - w // 10, w + w // 10)
    h = randint(h - h // 10, h + h // 10)

    return image.resize((w, h))


def __noise(image: Image) -> Image:
    return image.filter(ImageFilter.MedianFilter)


class ComparatorDataloader:
    def __init__(
            self,
            dataset: InstanceImage,
            mapping_images: str or Path,
            processing_rate: float,
            batch_size: int
    ):
        self.__dataset = dataset
        self.__mapping_images = Path(mapping_images)
        self.__batch_size = batch_size
        self.__processing_rate = processing_rate

        self.__suffix = 'png'

        self.__data_ids = _chunks(list(range(len(self.__dataset))), self.__batch_size)

        last = -1
        if os.path.exists(self.__mapping_images):
            for entry in self.__mapping_images.iterdir():
                try:
                    current = int(entry.name.removesuffix(entry.suffix))
                    last = max(last, current)
                except Exception:
                    pass

        self.__mapping_images.mkdir(parents=True, exist_ok=True)
        if last != len(self.__dataset) - 1:
            print(f'Generate mapping images')
            for i in tqdm(range(last + 1, len(self.__dataset))):
                image = self.__dataset[i]
                mapping_image = _random_transform(image, self.__processing_rate)
                mapping_image.save(self.__mapping_images.joinpath(f'{i}.{self.__suffix}'), self.__suffix.upper())

    def __len__(self):
        return len(self.__data_ids)

    def __getitem__(self, idx):
        ids = self.__data_ids[idx]
        keys = [self.__dataset[id] for id in ids]

        mapping_images = [self.__load_mapping_image(id) for id in ids]
        queries = mapping_images.copy()
        shuffle(queries)

        labels = []
        for origin in mapping_images:
            label = []
            for image in queries:
                if origin == image:
                    label.append(1.0)
                else:
                    label.append(0.0)
            labels.append(label)

        labels = torch.tensor(labels)

        return keys, queries, labels

    def shuffle(self):
        data_ids = list(range(len(self.__dataset)))
        shuffle(data_ids)

        self.__data_ids = _chunks(data_ids, self.__batch_size)

    def __load_mapping_image(self, id):
        path = self.__mapping_images.joinpath(f'{id}.{self.__suffix}')
        image = Image.open(path).convert('RGB')

        return image
