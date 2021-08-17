import os
from pathlib import Path

from PIL.Image import Image
from torch.utils.data import Dataset

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

        self.__local.mkdir(parents=True, exist_ok=True)

        self.__suffix = 'png'

        if not os.path.exists(self.__local):
            counter = 0
            for image, boxes in self.__coco:
                for box in boxes:
                    instance_image = image.crop(box)
                    instance_image.save(self.__local.joinpath(f'{counter}.{self.__suffix}', self.__suffix.upper()))
                    counter += 1

        self.__data_size = 0
        for entry in self.__local.iterdir():
            if entry.suffix == self.__suffix and _represents_int(entry.name):
                self.__data_size += 1

    def __len__(self):
        return self.__data_size

    def __getitem__(self, idx):
        path = self.__local.joinpath(f'{idx}.{self.__suffix}')
        image = Image.open(path).convert('RGB')

        return image


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')

    coco = COCO(
        remote='http://images.cocodataset.org/zips/train2017.zip',
        local=data_path.joinpath('train2017'),
        annotation_remote='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        annotation_local=data_path.joinpath('anotations')
    )

    instance_image = InstanceImage(
        coco=coco,
        local=data_path.joinpath('instances')
    )

