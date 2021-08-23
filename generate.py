import argparse
import os
from pathlib import Path

from src.data.dataset import COCODataset, InstanceDataset
from src.data.gernerator import BoundingBoxImageGenerator, NoisedImageGenerator


def generate(
        origin_path: str or Path,
        path: str or Path,
        dataset: str,
        expand: int,
        noise: float,
        force: bool = False
):
    origin_path = Path(origin_path)
    path = Path(path)

    coco = COCODataset(
        path=origin_path,
        dataset=dataset
    )

    bounding_box_image_generator = BoundingBoxImageGenerator(
        dataset=coco,
        path=path,
        format='jpg'
    )

    bounding_box_image_generator.generate(force=force)

    instance_dataset = InstanceDataset(
        path=path,
        dataset=dataset
    )

    noised_image_generator = NoisedImageGenerator(
        dataset=instance_dataset,
        format='jpg'
    )

    for i in range(expand):
        noised_image_generator.generate(noise)


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent

    origin_path = root_path.joinpath('data').joinpath('coco')
    date_path = root_path.joinpath('data').joinpath('instances')

    parser = argparse.ArgumentParser()

    parser.add_argument('--origin_path', type=str, default=str(origin_path))
    parser.add_argument('--path', type=str, default=str(date_path))
    parser.add_argument('--dataset', type=str, default='train2017')
    parser.add_argument('--expand', type=int, default=4)
    parser.add_argument('--noise', type=float, default=0.4)
    parser.add_argument('--force', type=bool, default=False)

    args = parser.parse_args()

    generate(args.origin_path, args.path, args.dataset, args.force)
