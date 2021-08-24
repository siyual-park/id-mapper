import abc
import math
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from src.data.dataset import CompareDataset
from src.model.checkpoint import HardCheckpoint, SoftCheckpoint
from src.model.comparator import Comparator
from src.optimiser.lookahead import Lookahead
from src.optimiser.radam import RAdam


class Trainer:
    def __init__(
            self,
            checkpoint: str or Path,
            model: nn.Module,
            optimizer: Optimizer,
    ):
        checkpoint = Path(checkpoint)

        best_checkpoint_path = checkpoint.joinpath('best.pt')
        last_checkpoint_path = checkpoint.joinpath('last.pt')

        self.__model = model
        self.__optimizer = optimizer

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__model.to(self.__device)

        self.__best_checkpoint = HardCheckpoint(
            path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

        self.__last_checkpoint = HardCheckpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

        self.__in_memory_checkpoint = SoftCheckpoint(
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=float('inf')
        )

    async def run(self, epochs: int) -> None:
        self.load()
        await self.sync_best_checkpoint()

        print(
            'Training start. Final epochs is {:3d}, pre best loss is {:5.2f}.'.format(
                epochs,
                self.__best_checkpoint.loss
            ),
            flush=True
        )

        start_time = time()

        for epoch in range(self.__last_checkpoint.epoch + 1, epochs + 1):
            self.__last_checkpoint.epoch = epoch

            epoch_start_time = time()

            self.__last_checkpoint.loss = await self.train()
            self.__last_checkpoint.loss = await self.evaluate()

            epoch_end_time = time()

            print(
                '{:3d} epoch, {:5.2f} loss, {:8.2f} ppl, {:5.2f}s'.format(
                    epoch,
                    self.__last_checkpoint.loss,
                    math.exp(self.__last_checkpoint.loss),
                    (epoch_end_time - epoch_start_time),
                ),
                flush=True
            )

            self.save()

        end_time = time()

        print('Training finish.', flush=True)
        print(
            '{:3d} epoch, {:5.2f} valid loss, {:8.2f} ppl, {:5.2f}s'.format(
                self.__best_checkpoint.epoch,
                self.__best_checkpoint.loss,
                math.exp(self.__best_checkpoint.loss),
                (end_time - start_time),
            ),
        )

    def load(self):
        self.__in_memory_checkpoint.save()
        self.__best_checkpoint.load(map_location=self.__device)
        self.__in_memory_checkpoint.load(map_location=self.__device)

        self.__last_checkpoint.load(map_location=self.__device)

    def save(self):
        self.__last_checkpoint.save()

        if self.__best_checkpoint.loss >= self.__last_checkpoint.loss:
            self.__best_checkpoint.loss = self.__last_checkpoint.loss
            self.__best_checkpoint.epoch = self.__last_checkpoint.epoch

            self.__best_checkpoint.save()

    async def sync_best_checkpoint(self):
        self.__in_memory_checkpoint.save()

        self.__best_checkpoint.load(map_location=self.__device)
        self.__best_checkpoint.loss = await self.evaluate()

        self.__in_memory_checkpoint.load(map_location=self.__device)

    @abc.abstractmethod
    async def train(self) -> float:
        raise NotImplemented

    @abc.abstractmethod
    async def evaluate(self) -> float:
        raise NotImplemented


class ComparatorTrainer(Trainer):
    def __init__(
            self,
            checkpoint: str or Path,
            model: Comparator,
            train_dataset: CompareDataset,
            val_dataset: CompareDataset,
            lr: float,
            k: int,
            alpha: float
    ):
        optimizer = RAdam(model.parameters(), lr=lr)
        optimizer = Lookahead(optimizer, k=k, alpha=alpha)

        super().__init__(
            checkpoint,
            model,
            optimizer
        )

        self.__train_dataset = train_dataset
        self.__val_dataset = val_dataset

        self.__criterion = nn.BCEWithLogitsLoss()
        self.__criterion.to(self.__device)

    async def train(self) -> float:
        self.__train_dataset.shuffle()

        total_loss = 0.0

        data = tqdm(self.__train_dataset)
        for i, (keys, queries, labels) in enumerate(data):
            keys = keys.to(self.__device)
            queries = queries.to(self.__device)
            labels = labels.to(self.__device)

            self.__optimizer.zero_grad()

            result = self.__model(keys, queries)

            loss = self.__criterion(result, labels)
            loss.backward()

            total_loss += loss.item()

            self.__optimizer.step()

            cur_loss = total_loss / (i + 1)
            data.set_description('{:3d} epoch, {:5.2f} loss, {:8.2f} ppl'.format(
                self.__last_checkpoint.epoch,
                cur_loss,
                math.exp(cur_loss)
            ))

        return total_loss / len(data)

    async def evaluate(self) -> float:
        self.__val_dataset.shuffle()

        total_loss = 0.0

        for keys, queries, labels in tqdm(self.__val_dataset):
            keys = keys.to(self.__device)
            queries = queries.to(self.__device)
            labels = labels.to(self.__device)

            result = self.__model(
                keys=keys,
                queries=queries
            )

            loss = self.__criterion(result, labels)
            total_loss += loss.item()

        return total_loss / len(self.__val_dataset)
