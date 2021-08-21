import abc
import asyncio
import collections
import math
from pathlib import Path
from time import time
from typing import TypeVar

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

T = TypeVar('T')


class Trainer:
    def __init__(
            self,
            model_dict_path: str or Path,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: _Loss,
            val_criterion: _Loss
    ):
        self.__model = model
        self.__optimizer = optimizer
        self.__criterion = criterion
        self.__val_criterion = val_criterion

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__model.to(self.__device)
        self.__criterion.to(self.__device)
        self.__val_criterion.to(self.__device)

        self.__epoch = 0

        self.__best_loss = float('inf')
        self.__best_epoch = 0

        model_dict_path = Path(model_dict_path)
        self.__best_model_path = Path('{}/best.pt'.format(model_dict_path))
        self.__last_model_path = Path('{}/last.pt'.format(model_dict_path))

        model_dict_path.parent.mkdir(parents=True, exist_ok=True)
        self.__best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.__last_model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.__best_model_path.exists():
            origin_state_dict = collections.OrderedDict(self.__model.state_dict())

            self.__load(self.__best_model_path, load_optimizer=False)

            loop = asyncio.get_event_loop()
            self.__best_loss = loop.run_until_complete(self.__evaluate())
            self.__best_epoch = self.__epoch

            self.__epoch = 0
            self.__model.load_state_dict(origin_state_dict)

            self.__log('Load best model. Best loss is {:5.2f}.'.format(self.__best_loss))

        if self.__last_model_path.exists():
            self.__load(self.__last_model_path, load_optimizer=True)
            self.__log('Load training model. Training proceeded {:3d} epoch.'.format(self.__epoch))

    async def run(self, epochs: int) -> None:
        self.__log('Training start. Final epochs is {:3d}.'.format(epochs))
        start_time = time()

        for epoch in range(self.__epoch + 1, epochs + 1):
            self.__epoch = epoch

            epoch_start_time = time()

            await self.__train()
            val_loss = await self.__evaluate()

            epoch_end_time = time()

            self.__log(
                '{:3d} epoch, {:5.2f} valid loss, {:8.2f} valid ppl, {:5.2f}s'.format(
                    epoch,
                    val_loss,
                    math.exp(val_loss),
                    (epoch_end_time - epoch_start_time),
                )
            )

            if val_loss <= self.__best_loss:
                self.__best_epoch = self.__epoch
                self.__best_loss = val_loss
                self.__save(self.__best_model_path)
            self.__save(self.__last_model_path)

        end_time = time()

        self.__log(
            '{:3d} best epoch, {:5.2f} best valid loss, {:8.2f} best valid ppl, {:5.2f}s total'.format(
                self.__best_epoch,
                self.__best_loss,
                math.exp(self.__best_loss),
                (end_time - start_time),
            ),
        )

    @abc.abstractmethod
    async def __train(self) -> None:
        raise NotImplemented

    @abc.abstractmethod
    async def __evaluate(self) -> float:
        raise NotImplemented

    def __load(self, path, load_optimizer: bool):
        checkpoint = torch.load(path, map_location=self.__device)

        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']

        self.__model.load_state_dict(model_state_dict)
        if load_optimizer:
            self.__optimizer.load_state_dict(optimizer_state_dict)
        self.__epoch = epoch

    def __save(self, path):
        torch.save(
            {
                'epoch': self.__epoch,
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
            },
            path
        )

    def __log(self, message: str) -> None:
        print(message)
