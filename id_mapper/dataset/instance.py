import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from id_mapper.dataset.coco import COCO


def _represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class InstanceImage(Dataset):
    def __init__(
            self,
            coco: COCO,
            local: Path or str
    ):
        self.__local = Path(local)
        self.__coco = coco

        self.__suffix = 'png'

        if not os.path.exists(self.__local):
            self.__local.mkdir(parents=True, exist_ok=True)

            print(f'Generate dataset')
            counter = 0
            for image, boxes in tqdm(self.__coco):
                for box in boxes:
                    instance_image = image.crop(box)
                    instance_image.save(self.__local.joinpath(f'{counter}.{self.__suffix}'), self.__suffix.upper())
                    counter += 1

        self.__data_size = 0
        for entry in self.__local.iterdir():
            if entry.suffix == f'.{self.__suffix}' and _represents_int(entry.name.removesuffix(entry.suffix)):
                self.__data_size += 1

    def __len__(self):
        return self.__data_size

    def __getitem__(self, idx):
        path = self.__local.joinpath(f'{idx}.{self.__suffix}')
        image = Image.open(path).convert('RGB')

        return image
