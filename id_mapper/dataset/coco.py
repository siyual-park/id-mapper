import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib import request

import numpy as np
from PIL.Image import Image
from pycocotools import coco
from torch.utils.data import Dataset
from tqdm import tqdm


def _load(remote: str, local: Path, cache: Path) -> None:
    if not os.path.exists(local):
        if not os.path.exists(cache):
            print(f'Download dataset from {remote} to {cache}')
            _download(remote, cache)

        print(f'Unzip dataset from {cache} to {local}')
        _unzip(cache, local)

        print(f'Remove cached dataset from {cache}')
        os.remove(cache)


def _download(remote: str, local: Path) -> None:
    def download_progress_hook(progress_bar):
        last_block = [0]

        def update_to(count=1, block_size=1, total_size=None):
            if total_size is not None:
                progress_bar.total = total_size
            progress_bar.update((count - last_block[0]) * block_size)
            last_block[0] = count

        return update_to

    with tqdm() as t:
        hook = download_progress_hook(t)
        request.urlretrieve(remote, local, reporthook=hook)


def _unzip(origin: Path, to: Path) -> None:
    with zipfile.ZipFile(origin, 'r') as zip_ref:
        zip_ref.extractall(to)


class COCO(Dataset):
    def __init__(
            self,
            remote: Optional[str],
            local: Path or str,
            annotation_remote: Optional[str],
            annotation_local: Path or str
    ):
        self.__remote = remote
        self.__annotation_remote = annotation_remote

        self.__local = Path(local)
        self.__annotation_local = Path(annotation_local)

        self.__local_zip = Path(local).with_suffix('.zip')
        self.__annotation_local_zip = Path(annotation_local).with_suffix('.zip')

        self.__local.parent.mkdir(parents=True, exist_ok=True)
        self.__annotation_local.parent.mkdir(parents=True, exist_ok=True)
        self.__local_zip.parent.mkdir(parents=True, exist_ok=True)
        self.__annotation_local_zip.parent.mkdir(parents=True, exist_ok=True)

        _load(self.__remote, self.__local, self.__local_zip)
        _load(self.__annotation_remote, self.__annotation_local, self.__annotation_local_zip)

        self.__coco = coco.COCO(self.__annotation_local.joinpath('instances_' + self.__local.name + '.json'))

        whole_image_ids = self.__coco.getImgIds()  # original length of train2017 is 118287

        self.__image_ids = []
        # to remove not annotated image idx
        self.__no_anno_list = []

        for idx in whole_image_ids:
            annotations_ids = self.__coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0:
                self.__no_anno_list.append(idx)
            else:
                self.__image_ids.append(idx)

        # load class names (name -> label)
        categories = self.__coco.loadCats(self.__coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.__classes = {}
        self.__coco_labels = {}
        self.__coco_labels_inverse = {}
        for c in categories:
            self.__coco_labels[len(self.__classes)] = c['id']
            self.__coco_labels_inverse[c['id']] = len(self.__classes)
            self.__classes[c['name']] = len(self.__classes)

        # also load the reverse (label -> name)
        self.__labels = {}
        for key, value in self.__classes.items():
            self.__labels[value] = key

    def __len__(self):
        return len(self.__image_ids)

    def __getitem__(self, idx):
        image, (w, h) = self.load_image(idx)
        annotations = self.load_annotations(idx)

        boxes = annotations[:, :4]
        labels = annotations[:, 4]

        return image, boxes

    def load_image(self, image_index) -> Tuple[Image, Tuple[int, int]]:
        image_info = self.__coco.loadImgs(self.__image_ids[image_index])[0]
        path = self.__local.joinpath(image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image, (image_info['width'], image_info['height'])

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.__coco.getAnnIds(imgIds=self.__image_ids[image_index], iscrowd=False)
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
            annotation[0, 4] = self.__coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def __coco_label_to_label(self, coco_label):
        return self.__coco_labels_inverse[coco_label]
