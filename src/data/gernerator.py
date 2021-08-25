import os
from pathlib import Path
from random import random, randint

from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm

from src.data.dataset import COCODataset, InstanceDataset
from src.data.utils import get_data_size


class BoundingBoxImageGenerator:
    def __init__(
            self,
            dataset: COCODataset,
            path: str or Path,
            format: str
    ):
        path = Path(path)

        self.__dataset = dataset
        self.__path = path.joinpath(dataset.dataset)
        self.__format = format

    def generate(self, force: bool = False):
        if os.path.exists(self.__path) and force:
            os.remove(self.__path)

        self.__path.mkdir(parents=True, exist_ok=True)

        existed_data_size = get_data_size(self.__path)
        current_data_index = 0

        print(f'Generate bounding box images from {self.__dataset.data_path} to {self.__path}')
        for image, annotations in tqdm(self.__dataset):
            boxes = annotations[:, :4]
            for box in boxes:
                current_data_index += 1

                if current_data_index <= existed_data_size:
                    continue

                image_dir = self.__path.joinpath(str(current_data_index - 1))
                image_dir.mkdir(parents=True)

                instance_image = image.crop(box)
                instance_image.save(image_dir.joinpath(f'0.{self.__format}'))


class NoisedImageGenerator:
    def __init__(
            self,
            dataset: InstanceDataset,
            format: str
    ):
        self.__dataset = dataset
        self.__format = format

    def generate(self, rate: float):
        print(f'Generate noised images')

        for images, images_path in tqdm(self.__dataset):
            if len(images) == 0:
                continue

            origin_image: Image.Image = images[0]
            noised_image = self.__noise(origin_image, rate)

            noised_image_id = len(images)
            noised_image.save(images_path.joinpath(f'{noised_image_id}.{self.__format}'))

    def __noise(self, image: Image.Image, rate: float) -> Image.Image:
        current_rate = random()
        if current_rate <= rate:
            image = self.__rotate(image)

        current_rate = random()
        if current_rate <= rate:
            image = self.__crop(image)

        current_rate = random()
        if current_rate <= rate:
            image = self.__filter(image)

        current_rate = random()
        if current_rate <= rate:
            image = self.__resize(image)

        return image

    def __rotate(self, image: Image.Image) -> Image.Image:
        angle = randint(-10, 10)
        return image.rotate(angle)

    def __crop(self, image: Image.Image) -> Image.Image:
        (w, h) = image.size

        x1 = randint(0, w // 10)
        y1 = randint(0, h // 10)
        x2 = randint(w - w // 10, w)
        y2 = randint(h - h // 10, h)

        image = image.crop((x1, y1, x2, y2))
        image = image.resize((w, h))

        return image

    def __resize(self, image: Image.Image) -> Image.Image:
        (w, h) = image.size

        w = randint(w - w // 10, w + w // 10)
        h = randint(h - h // 10, h + h // 10)

        return image.resize((w, h))

    def __filter(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.MedianFilter)
