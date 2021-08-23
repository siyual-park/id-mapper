import os
import zipfile
from pathlib import Path
from urllib import request

from tqdm import tqdm


class Downloader:
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
        super().__init__(source, local)

        local = Path(local)
        filename = get_remote_filename(source)
        download_tmp = local.parent.joinpath(filename)

        self.__downloader = Downloader(
            source=source,
            local=download_tmp
        )
        self.local = local

    def download(self, force: bool = False) -> None:
        self.__downloader.download(force=force)

    def unzip(self, force: bool = False):
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
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

        super().__init__(source, local.joinpath(dataset))

        self.__downloader = ZIPDownloader(
            source=source,
            local=local
        )
        self.coco_local = local

    def download(self, force: bool = False) -> None:
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
                return

        self.__downloader.download(force=force)
        self.__downloader.unzip(force=False)
        self.__downloader.clear(all=False)


class COCOAnnotationDownloader(Downloader):
    def __init__(self, source: str, local: str or Path):
        local = Path(local)

        super().__init__(source, local.joinpath('annotations'))

        self.__downloader = ZIPDownloader(
            source=source,
            local=local
        )
        self.coco_local = local

    def download(self, force: bool = False) -> None:
        if os.path.exists(self.local):
            if force:
                os.remove(self.local)
            else:
                return

        self.__downloader.download(force=force)
        self.__downloader.unzip(force=False)
        self.__downloader.clear(all=False)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')
    checkpoint_path = root_path.joinpath('checkpoint')

    train_downloader = COCOImageDownloader(
        source='http://images.cocodataset.org/zips/train2017.zip',
        local=data_path.joinpath('coco')
    )
    val_downloader = COCOImageDownloader(
        source='http://images.cocodataset.org/zips/val2017.zip',
        local=data_path.joinpath('coco')
    )
    annotation_downloader = COCOAnnotationDownloader(
        source='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        local=data_path.joinpath('coco')
    )

    val_downloader.download()
    train_downloader.download()
