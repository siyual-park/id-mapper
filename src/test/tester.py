import math
from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import torch
from pandas import DataFrame
from torch import nn
from tqdm import tqdm

from src.data.dataloader import CompareDataLoader
from src.model.comparator import Comparator
from src.train.checkpoint import HardCheckpoint, SoftCheckpoint


class Tester:
    def __init__(
            self,
            checkpoint: str or Path,
            model: nn.Module
    ):
        checkpoint = Path(checkpoint)

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

        loss, pre_time = await self.evaluate()

        self._in_memory_checkpoint.load(map_location=self._device)

        print(
            '{:3d} epoch, {:5.2f} loss, {:8.2f} ppl, {:5.5f}s/it'.format(
                self._checkpoint.epoch,
                loss,
                math.exp(loss),
                pre_time,
            ),
        )

    async def evaluate(self) -> Tuple[float, float]:
        raise NotImplemented


class ComparatorTester(Tester):
    def __init__(
            self,
            checkpoint: str or Path,
            model: Comparator,
            dataset: CompareDataLoader,
    ):

        super().__init__(
            checkpoint,
            model
        )

        self.__dataset = dataset

        self.__criterion = nn.BCELoss()
        self.__criterion.to(self._device)

    async def evaluate(self) -> Tuple[float, float]:
        self.__dataset.shuffle()

        total_loss = 0.0
        total_time = 0.0

        confusion_matrix = np.zeros((2, 2))

        for keys, queries, expected in tqdm(self.__dataset):
            keys = keys.to(self._device)
            queries = queries.to(self._device)
            expected = expected.to(self._device)

            start = time()
            actual = self._model(keys, queries)
            end = time()

            loss = self.__criterion(actual, expected)
            total_loss += loss.item()
            total_time += start - end

            confusion_matrix += self.get_confusion_matrix(actual, expected)

        confusion_matrix /= confusion_matrix.sum()
        print('Confusion matrix')
        print(DataFrame(confusion_matrix))

        precision = self.get_precision(confusion_matrix)
        recall = self.get_precision(confusion_matrix)
        f1 = self.get_f1(precision, recall)

        print(
            '{:5.2f} f1, {:5.2f} precision, {:5.2f} recall;'.format(
                f1,
                precision,
                recall,
            ),
        )

        return total_loss / len(self.__dataset), total_time / len(self.__dataset)

    def get_confusion_matrix(self, actual: torch.Tensor, expected: torch.Tensor):
        matrix = np.zeros((2, 2))

        actual = actual.cpu().detach().numpy()
        expected = expected.cpu().detach().numpy()

        actual = np.where(actual > 0.5, 1, 0)

        (x, y) = actual.shape
        for i in range(x):
            for j in range(y):
                matrix[int(actual[i, j]), int(expected[i, j])] += 1

        return matrix

    def get_f1(self, precision, recall):
        return 2 * (recall * precision) / (recall + precision)

    def get_precision(self, confusion_matrix):
        return confusion_matrix[1, 1] / confusion_matrix[1, :].sum()

    def get_recall(self, confusion_matrix):
        return confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
