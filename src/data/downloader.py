import os
import zipfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List
from urllib import request

from tqdm import tqdm


class Downloader(metaclass=ABCMeta):
    @abstractmethod
    def download(self, force: bool = False) -> None:
        pass


class DefaultDownloader(Downloader):
    def __init__(self, source: str, local: str or Path):
        self.source = source
        self.local = Path(local)

    def download(self, force: bool = False) -> None:
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
                return

        print(f'Download from {self.source} to {self.local}')

        self.local.parent.mkdir(parents=True, exist_ok=True)

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
            request.urlretrieve(self.source, self.local, reporthook=hook)


def get_remote_filename(source: str) -> str:
    tokens = source.split('/')
    return tokens[len(tokens) - 1]


class ZIPDownloader(Downloader):
    def __init__(self, source: str, local: str or Path):
        local = Path(local)
        filename = get_remote_filename(source)
        download_tmp = local.parent.joinpath(filename)

        self.source = source
        self.local = local

        self.__downloader = DefaultDownloader(
            source=source,
            local=download_tmp
        )

    def download(self, force: bool = False) -> None:
        self.__downloader.download(force=force)

    def unzip(self, override: bool = False):
        if not override and os.path.exists(self.local):
            return

        with zipfile.ZipFile(self.__downloader.local, 'r') as zip_ref:
            zip_ref.extractall(self.local)

    def clear(self, all: bool = False):
        if os.path.exists(self.__downloader.local):
            os.remove(self.__downloader.local)

        if not all:
            return

        if os.path.exists(self.local):
            os.remove(self.local)


class COCOImageDownloader(Downloader):
    def __init__(self, source: str, local: str or Path):
        local = Path(local)
        filename = get_remote_filename(source)
        dataset = os.path.splitext(filename)[0]

        self.source = source
        self.local = local.joinpath(dataset)
        self.coco_local = local

        self.__downloader = ZIPDownloader(
            source=source,
            local=self.coco_local
        )

    def download(self, force: bool = False) -> None:
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
                return

        self.__downloader.download(force=force)
        self.__downloader.unzip(override=True)
        self.__downloader.clear(all=False)


class COCOAnnotationDownloader(Downloader):
    def __init__(self, sources: List[str], local: str or Path):
        local = Path(local)

        self.sources = sources
        self.local = local.joinpath('annotations')
        self.coco_local = local

    def download(self, force: bool = False) -> None:
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
                return

        for source in self.sources:
            downloader = ZIPDownloader(
                source=source,
                local=self.coco_local
            )

            downloader.download(force=force)
            downloader.unzip(override=True)
            downloader.clear(all=False)
