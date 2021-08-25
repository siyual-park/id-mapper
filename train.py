import argparse
import asyncio
import os
from pathlib import Path
from time import time

from src.data.dataset import LocalInstanceDataset, CompareDataset
from src.model.comparator import Comparator
from src.model.tokenizer import Tokenizer
from src.train.trainer import ComparatorTrainer

if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))

    root_path = path.parent
    root_parent_path = path.parent.parent

    data_path = root_parent_path.joinpath('data')
    checkpoints_path = root_path.joinpath('checkpoints')

    coco_data_path = data_path.joinpath('coco')
    instances_date_path = data_path.joinpath('instances')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default=str(time()))

    parser.add_argument('--train', type=str, default='train2017')
    parser.add_argument('--val', type=str, default='val2017')

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--token_size', type=int, default=1024)
    parser.add_argument('--deep', type=int, default=2)
    parser.add_argument('--res_block_deep', type=int, default=2)
    parser.add_argument('--dropout_prob', type=float, default=0.4)

    args = parser.parse_args()

    train_dataset = LocalInstanceDataset(
        path=instances_date_path,
        dataset=args.train
    )
    val_dataset = LocalInstanceDataset(
        path=instances_date_path,
        dataset=args.val
    )

    train_dataset = CompareDataset(
        dataset=train_dataset,
        image_size=args.image_size
    )
    val_dataset = CompareDataset(
        dataset=val_dataset,
        image_size=args.image_size
    )

    tokenizer = Tokenizer(
        image_size=args.image_size,
        token_size=args.token_size,
        deep=args.deep,
        res_block_deep=args.res_block_deep,
        dropout_prob=args.dropout_prob
    )
    compare = Comparator(tokenizer=tokenizer)

    trainer = ComparatorTrainer(
        checkpoint=checkpoints_path.joinpath(args.checkpoint),
        model=compare,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        k=args.k,
        alpha=args.alpha
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.run(epochs=args.epochs))

