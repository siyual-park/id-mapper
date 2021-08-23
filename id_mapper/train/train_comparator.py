import argparse
import asyncio
import math
import os
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from id_mapper.dataloader.comparator import ComparatorDataloader
from id_mapper.dataset.coco import COCO
from id_mapper.dataset.instance import InstanceImage
from id_mapper.model.comparator import Comparator
from id_mapper.model.tokenizer import Tokenizer
from id_mapper.optimizer.lookahead import Lookahead
from id_mapper.optimizer.radam import RAdam
from id_mapper.train import trainer


class Trainer(trainer.Trainer):
    def __init__(
            self,
            model: Comparator,
            checkpoint: str or Path,
            train_dataset: InstanceImage,
            val_dataset: InstanceImage,
            processing_rate: float,
            batch_size: int,
            lr: float
    ):
        optimizer = RAdam(model.parameters(), lr=lr)
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

        def mapping_images(path: Path) -> Path:
            return path.parent.joinpath(f'{path.name}_mapping{path.suffix}')

        train_data_loader = ComparatorDataloader(
            dataset=train_dataset,
            mapping_images=mapping_images(train_dataset.local),
            processing_rate=processing_rate,
            batch_size=batch_size
        )
        val_data_loader = ComparatorDataloader(
            dataset=val_dataset,
            mapping_images=mapping_images(val_dataset.local),
            processing_rate=processing_rate,
            batch_size=batch_size
        )

        self.__train_data_loader = train_data_loader
        self.__val_data_loader = val_data_loader

        self.__batch_size = batch_size

        super().__init__(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([batch_size - 1] * batch_size))
        )

    async def __evaluate(self) -> float:
        self.__val_data_loader.shuffle()

        self.__model.eval()

        total_loss = 0.0
        progressed = 0

        with torch.no_grad():
            for keys, queries, labels in self.__val_data_loader:
                if labels.size(0) != self.__batch_size:
                    continue

                result = self.__model(
                    keys=keys,
                    queries=queries
                )

                labels = labels.to(self.__device)

                loss = self.__criterion(result, labels)
                total_loss += loss.item()
                progressed += 1

        return total_loss / progressed

    async def __train(self) -> None:
        self.__train_data_loader.shuffle()

        self.__model.train()

        total_loss = 0.0
        progressed = 0

        train_data = tqdm(self.__train_data_loader)
        for i, (keys, queries, labels) in enumerate(train_data):
            if labels.size(0) != self.__batch_size:
                continue

            self.__optimizer.zero_grad()

            result = self.__model(
                keys=keys,
                queries=queries
            )

            labels = labels.to(self.__device)

            loss = self.__criterion(result, labels)
            loss.backward()

            total_loss += loss.item()
            progressed += 1

            self.__optimizer.step()

            cur_loss = total_loss / progressed
            train_data.set_description('{:3d} epoch, {:5.2f} loss, {:8.2f} ppl'.format(
                self.__epoch,
                cur_loss,
                math.exp(cur_loss)
            ))


if __name__ == '__main__':
    path = Path(os.path.abspath(__file__))
    root_path = path.parent.parent.parent

    data_path = root_path.joinpath('data')
    checkpoint_path = root_path.joinpath('checkpoint')

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default='train2017')
    parser.add_argument(
        '--train_remote',
        type=str,
        default='http://images.cocodataset.org/zips/train2017.zip',
    )

    parser.add_argument('--val', type=str, default='val2017')
    parser.add_argument(
        '--val_remote',
        type=str,
        default='http://images.cocodataset.org/zips/val2017.zip',
    )

    parser.add_argument('--annotations', type=str, default='annotations')
    parser.add_argument(
        '--annotations_remote',
        type=str,
        default='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    )

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--checkpoint', type=str, default='comparator')

    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--token_size', type=int, default=1024)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--processing_rate', type=float, default=0.3)

    args = parser.parse_args()

    train_coco = COCO(
        remote=args.train_remote,
        local=data_path.joinpath(args.train),
        annotation_remote=args.annotations_remote,
        annotation_local=data_path.joinpath(args.annotations)
    )
    val_coco = COCO(
        remote=args.val_remote,
        local=data_path.joinpath(args.val),
        annotation_remote=args.annotations_remote,
        annotation_local=data_path.joinpath(args.annotations)
    )

    train_instance_image = InstanceImage(
        coco=train_coco,
        local=data_path.joinpath('train_instances')
    )
    val_instance_image = InstanceImage(
        coco=val_coco,
        local=data_path.joinpath('val_instances')
    )

    tokenizer = Tokenizer(
        image_size=args.image_size,
        token_size=args.token_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    )

    model = Comparator(
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        checkpoint=checkpoint_path.joinpath(args.checkpoint),
        train_dataset=train_instance_image,
        val_dataset=val_instance_image,
        processing_rate=args.processing_rate,
        batch_size=args.batch_size,
        lr=args.lr
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.run(epochs=args.epochs))
