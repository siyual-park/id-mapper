from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


class Checkpoint(metaclass=ABCMeta):
    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @model.setter
    @abstractmethod
    def model(self, value: nn.Module) -> None:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        pass

    @optimizer.setter
    @abstractmethod
    def optimizer(self, value: Optimizer) -> None:
        pass

    @property
    @abstractmethod
    def epoch(self) -> int:
        pass

    @epoch.setter
    @abstractmethod
    def epoch(self, value: Optimizer) -> None:
        pass

    @property
    @abstractmethod
    def loss(self) -> float:
        pass

    @loss.setter
    @abstractmethod
    def loss(self, value: float) -> None:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, map_location=None):
        pass


class HardCheckpoint(Checkpoint):
    def __init__(
            self,
            path: str or Path,
            model: nn.Module,
            optimizer: Optimizer,
            epoch: int,
            loss: float
    ):
        self.__path = path

        self.__model = model
        self.__optimizer = optimizer
        self.__epoch = epoch
        self.__loss = loss

    @property
    def model(self) -> nn.Module:
        return self.__model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self.__model = value

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        self.__optimizer = value

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, value: Optimizer) -> None:
        self.__epoch = value

    @property
    def loss(self) -> float:
        return self.__loss

    @loss.setter
    def loss(self, value: float) -> None:
        self.__loss = value

    def save(self):
        torch.save(
            {
                'loss': self.loss,
                'epoch': self.epoch,
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
            },
            self.__path
        )

    def load(self, map_location=None):
        if not self.__path.exists():
            return

        checkpoint = torch.load(self.__path, map_location=map_location)

        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.__model.load_state_dict(model_state_dict)
        self.__optimizer.load_state_dict(optimizer_state_dict)
        self.__epoch = epoch
        self.__loss = loss


class SoftCheckpoint(Checkpoint):
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            epoch: int,
            loss: float
    ):
        self.__model = model
        self.__optimizer = optimizer
        self.__epoch = epoch
        self.__loss = loss

        self.__cached_model_state_dict = None
        self.__cached_optimizer_state_dict = None
        self.__cached_epoch = None
        self.__cached_loss = None

    @property
    def model(self) -> nn.Module:
        return self.__model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self.__model = value

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        self.__optimizer = value

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, value: Optimizer) -> None:
        self.__epoch = value

    @property
    def loss(self) -> float:
        return self.__loss

    @loss.setter
    def loss(self, value: float) -> None:
        self.__loss = value

    def save(self):
        self.__cached_model_state_dict = self.__model.state_dict()
        self.__cached_optimizer_state_dict = self.__optimizer.state_dict()
        self.__cached_epoch = self.epoch
        self.__cached_loss = self.loss

    def load(self, map_location=None):
        if self.__cached_model_state_dict is not None:
            self.__model.load_state_dict(self.__cached_model_state_dict)
        if self.__cached_optimizer_state_dict is not None:
            self.__optimizer.load_state_dict(self.__cached_optimizer_state_dict)
        if self.__cached_epoch is not None:
            self.__epoch = self.__cached_epoch
        if self.__cached_loss is not None:
            self.__loss = self.__cached_loss

        self.__cached_model_state_dict = None
        self.__cached_optimizer_state_dict = None
        self.__cached_epoch = None
        self.__cached_loss = None
