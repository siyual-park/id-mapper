import argparse
import os
from pathlib import Path
from typing import List

from src.data.downloader import COCOImageDownloader, COCOAnnotationDownloader


class DownloadInfo:
    def __init__(
            self,
            train: str,
            val: str,
            annotation: List[str]
    ):
        self.train = train
        self.val = val

        self.annotation = annotation


download_infos = {
    '2017': DownloadInfo(
        train='http://images.cocodataset.org/zips/train2017.zip',
        val='http://images.cocodataset.org/zips/val2017.zip',
        annotation=[
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        ]
    )
}


def download(dataset: str, local: str or Path, force: bool = False):
    download_info = download_infos[dataset]

    train_downloader = COCOImageDownloader(
        source=download_info.train,
        local=local
    )
    val_downloader = COCOImageDownloader(
        source=download_info.val,
        local=local
    )
    annotation_downloader = COCOAnnotationDownloader(
        sources=download_info.annotation,
        local=local
    )

    val_downloader.download(force=force)
    train_downloader.download(force=force)
    annotation_downloader.download(force=force)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent

    data_path = root_path.joinpath('data').joinpath('coco')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='2017')
    parser.add_argument('--force', type=bool, default=False)
    parser.add_argument('--path', type=str, default=str(data_path))

    args = parser.parse_args()

    dataset_path = data_path.joinpath(args.path)

    download(args.dataset, args.path, args.force)

