import argparse
import asyncio
import os
from pathlib import Path

from src.data.dataloader import CompareDataLoader
from src.data.dataset import LocalInstanceDataset
from src.model.comparator import Comparator
from src.model.tokenizer import Tokenizer
from src.test.tester import ComparatorTester

if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))

    root_path = path.parent
    root_parent_path = path.parent.parent

    data_path = root_parent_path.joinpath('data')
    checkpoints_path = root_path.joinpath('checkpoints')

    coco_data_path = data_path.joinpath('coco')
    instances_date_path = data_path.joinpath('instances')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='val2017')

    parser.add_argument('--epochs', type=int, default=40)

    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--image_set_num', type=int, default=2)
    parser.add_argument('--max_image_num', type=int, default=4)
    parser.add_argument('--token_size', type=int, default=256)
    parser.add_argument('--deep', type=int, default=1)
    parser.add_argument('--res_block_deep', type=int, default=2)

    args = parser.parse_args()

    dataset = LocalInstanceDataset(
        path=instances_date_path,
        dataset=args.dataset
    )

    dataset = CompareDataLoader(
        dataset=dataset,
        image_size=args.image_size,
        image_set_num=args.image_set_num,
        max_image_num=args.max_image_num,
    )

    tokenizer = Tokenizer(
        image_size=args.image_size,
        token_size=args.token_size,
        deep=args.deep,
        res_block_deep=args.res_block_deep,
        dropout_prob=0.0
    )
    compare = Comparator(tokenizer=tokenizer)

    trainer = ComparatorTester(
        checkpoint=checkpoints_path.joinpath(args.checkpoint),
        model=compare,
        dataset=dataset,
        epochs=args.epochs
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.run())

