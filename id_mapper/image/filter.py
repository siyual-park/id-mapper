from random import randint, random

from PIL.Image import Image
from PIL import ImageFilter


def random_transform(image: Image, rate: float) -> Image:
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
