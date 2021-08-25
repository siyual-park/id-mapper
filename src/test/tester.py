import math
from pathlib import Path
from time import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.data.dataset import CompareDataset
from src.model.comparator import Comparator
from src.train.checkpoint import HardCheckpoint, SoftCheckpoint


class Tester:
    def __init__(
            self,
            checkpoint: str or Path,
            model: nn.Module
    ):
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)

        self._model = model

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model.to(self._device)

        self._checkpoint = HardCheckpoint(
            path=checkpoint,
            model=model,
            optimizer=None,
            epoch=0,
            loss=float('inf')
        )

        self._in_memory_checkpoint = SoftCheckpoint(
            model=model,
            optimizer=None,
            epoch=0,
            loss=float('inf')
        )

    async def run(self) -> None:
        print(
            'epochs is {:3d}, train best loss is {:5.2f}.'.format(
                self._checkpoint.epoch,
                self._checkpoint.loss
            ),
            flush=True
        )

        self._in_memory_checkpoint.save()
        self._checkpoint.load(map_location=self._device)

        start_time = time()

        loss = await self.evaluate()

        end_time = time()

        self._in_memory_checkpoint.load(map_location=self._device)

        print(
            '{:3d} epoch, {:5.2f} loss, {:8.2f} ppl, {:5.2f}s'.format(
                self._checkpoint.epoch,
                loss,
                math.exp(loss),
                (end_time - start_time),
            ),
        )

    async def evaluate(self) -> float:
        raise NotImplemented


class ComparatorTester(Tester):
    def __init__(
            self,
            checkpoint: str or Path,
            model: Comparator,
            dataset: CompareDataset,
    ):

        super().__init__(
            checkpoint,
            model
        )

        self.__dataset = dataset

        self.__criterion = nn.BCELoss()
        self.__criterion.to(self._device)

    async def evaluate(self) -> float:
        self.__dataset.shuffle()

        total_loss = 0.0
        confusion_matrix = np.zeros((2, 2))

        for keys, queries, expected in tqdm(self.__dataset):
            keys = keys.to(self._device)
            queries = queries.to(self._device)
            expected = expected.to(self._device)

            actual = self._model(keys, queries)

            loss = self.__criterion(actual, expected)
            total_loss += loss.item()

            confusion_matrix += self.get_confusion_matrix(actual, expected)

        print(confusion_matrix, flush=True)
        return total_loss / len(self.__dataset)

    def get_confusion_matrix(self, actual: torch.Tensor, expected: torch.Tensor):
        matrix = np.zeros((2, 2))

        actual = actual.numpy()
        expected = expected.numpy()

        actual = np.where(actual > 0.5, 1, 0)

        (x, y) = actual.shape
        for i in range(x):
            for j in range(y):
                matrix[actual[i, j], expected[i, j]] += 1

        return matrix
