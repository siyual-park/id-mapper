import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request

from torch.utils.data import Dataset


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
    request.urlretrieve(remote, local)


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
        _load(self.__annotation_remote, self.__local, self.__local_zip)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')

    coco = COCO(
        remote='http://images.cocodataset.org/zips/train2017.zip',
        local=data_path.joinpath('val2017')
    )
