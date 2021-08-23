import argparse
import os
from pathlib import Path

from src.data.dataset import COCO
from src.data.gernerator import BoundingBoxImageGenerator


def generate(
        origin_path: str or Path,
        path: str or Path,
        dataset: str,
        force: bool = False
):
    origin_path = Path(origin_path)
    path = Path(path)

    coco = COCO(
        path=origin_path,
        dataset=dataset
    )

    bounding_box_image_generator = BoundingBoxImageGenerator(
        coco_dataset=coco,
        path=path,
        format='jpg'
    )

    bounding_box_image_generator.generate(force=force)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent

    origin_path = root_path.joinpath('data').joinpath('coco')
    date_path = root_path.joinpath('data').joinpath('instances')

    parser = argparse.ArgumentParser()

    parser.add_argument('--origin_path', type=str, default=str(origin_path))
    parser.add_argument('--path', type=str, default=str(date_path))
    parser.add_argument('--dataset', type=str, default='train2017')
    parser.add_argument('--force', type=bool, default=False)

    args = parser.parse_args()

    generate(args.origin_path, args.path, args.dataset, args.force)
