import os
import zipfile
from pathlib import Path
from urllib import request

from tqdm import tqdm


class Downloader:
    def __init__(self, source: str, target: str or Path):
        self.__source = source
        self.__target = Path(target)

    def download(self) -> None:
        if os.path.exists(self.__target):
            return

        print(f'Download from {self.__source} to {self.__target}')

        self.__target.parent.mkdir(parents=True, exist_ok=True)

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
            request.urlretrieve(self.__source, self.__target, reporthook=hook)


def get_remote_filename(source: str) -> str:
    tokens = source.split('/')
    return tokens[len(tokens) - 1]


class COCODownloader(Downloader):
    def __init__(self, source: str, target: str or Path):
        target = Path(target)
        filename = get_remote_filename(source)
        download_tmp = target.parent.joinpath(filename)

        super().__init__(source, download_tmp)

        dataset = os.path.splitext(filename)[0]

        self.local = target.joinpath(dataset)
        self.dataset = dataset

    def download(self) -> None:
        if os.path.exists(self.local):
            return

        self.local.parent.parent.mkdir(parents=True, exist_ok=True)

        super().download()

        with zipfile.ZipFile(self.__target, 'r') as zip_ref:
            zip_ref.extractall(self.local.parent)
        os.remove(self.__target)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')
    checkpoint_path = root_path.joinpath('checkpoint')

    downloader = COCODownloader(
        source='http://images.cocodataset.org/zips/val2017.zip',
        target=data_path.joinpath('coco')
    )

    downloader.download()
