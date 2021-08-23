from pathlib import Path
from random import shuffle

import numpy as np
from PIL import Image
from pycocotools import coco
from torch.utils import data

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

    def __getitem__(self, idx):
        id = self.__image_ids[idx]

        image = self.load_image(id)
        annotations = self.load_annotations(id)

        return image, annotations

    def load_image(self, image_id) -> Image.Image:
        image_info = self.__coco.loadImgs(image_id)[0]
        path = self.data_path.joinpath(image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image

    def load_annotations(self, image_id):
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

    def __getitem__(self, idx):
        id = self.__image_ids[idx]

        images_path = self.data_path.joinpath(str(id))

        images = []
        for entry in sorted(images_path.iterdir()):
            if represents_int(entry.name.removesuffix(entry.suffix)):
                image = Image.open(entry).convert('RGB')
                images.append(image)

        return images
