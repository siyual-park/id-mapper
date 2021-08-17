import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request

from torch.utils.data import Dataset


class COCO(Dataset):
    def __init__(
            self,
            remote: Optional[Path or str],
            local: Path or str
    ):
        self.__remote = remote

        self.__local = Path(local)
        self.__local_zip = Path(local).with_suffix('.zip')

        self.__local.parent.mkdir(parents=True, exist_ok=True)
        self.__local_zip.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.__local):
            if not os.path.exists(self.__local_zip):
                print(f'Download dataset from {self.__remote} to {self.__local_zip}')
                self.__download()

            print(f'Unzip dataset from {self.__local_zip} to {self.__local}')
            self.__unzip()

            print(f'Remove cached dataset from {self.__local_zip}')
            self.__remove_zip()

    def __download(self) -> None:
        if self.__remote is None:
            raise Exception()

        request.urlretrieve(self.__remote, self.__local_zip)

    def __unzip(self) -> None:
        with zipfile.ZipFile(self.__local_zip, 'r') as zip_ref:
            zip_ref.extractall(self.__local)

    def __remove_zip(self) -> None:
        os.remove(self.__local_zip)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')

    coco = COCO(
        remote='http://images.cocodataset.org/zips/train2017.zip',
        local=data_path.joinpath('val2017')
    )
