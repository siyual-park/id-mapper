from random import shuffle, random, randint, sample

import torch
from PIL.Image import Image
from PIL import ImageFilter

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
            processing_rate: float,
            batch_size: int
    ):
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__processing_rate = processing_rate

        self.__data_ids = _chunks(list(range(len(self.__dataset))), self.__batch_size)

    def shuffle(self):
        data_ids = list(range(len(self.__dataset)))
        shuffle(data_ids)

        self.__data_ids = _chunks(data_ids, self.__batch_size)

    def __len__(self):
        return len(self.__data_ids)

    def __getitem__(self, idx):
        ids = self.__data_ids[idx]
        keys = [self.__dataset[id] for id in ids]

        keys_size = len(keys)

        origin_queries = [_random_transform(image, self.__processing_rate) for image in keys]
        queries = sample(origin_queries, k=randint(keys_size - keys_size // 10, keys_size))
        shuffle(queries)

        labels = []
        for origin in origin_queries:
            label = []
            for image in queries:
                if origin == image:
                    label.append(1.0)
                else:
                    label.append(0.0)
            labels.append(label)

        labels = torch.tensor(labels)

        return keys, queries, labels
