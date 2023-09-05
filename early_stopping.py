from typing import Iterator
from torch import nn
from torch.optim import Optimizer
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Class that implements early stopping

        Args:
            patience (int, optional): The number of steps to wait
                without improvement before stopping the training. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored
                metric to qualify as an improvement.
                If the improvement is less than `min_delta`,
                it is not considered significant. Defaults to 0.0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_model_state_dict = None
        self.best_model_params = None
        self.best_optimizer_params = None
        self.min_test_loss = np.inf
        self.stop = False
        self.counter = 0

    def __call__(
        self, test_loss: float, model: nn.Module | any, optimizer: Optimizer
    ) -> None:
        """Checks if early stopping criteria are met.

        Args:
            test_loss (float): The current test loss.
            model (nn.Module | any): The model being trained.
            optimizer (Optimizer): The optimizer used for training.
        """
        if test_loss > (self.min_test_loss + self.min_delta):
            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True

        else:
            self.min_test_loss = test_loss
            self.best_model_state_dict = model.state_dict()
            self.best_model_params = model.parameters()
            self.best_optimizer_state_dict = optimizer.state_dict()
            self.counter = 0

    def restore_best(
        self, model: nn.Module | any, optimizer: Optimizer
    ) -> tuple[dict, dict, Iterator]:
        """Restores the best model and optimizer states.

        Args:
            model (nn.Module | any): The nmodel being trained.
            optimizer (Optimizer): The optimizer used for training.

        Returns:
            tuple[dict, dict, Iterator]: A tuple containing the best model state dictionary,
            best optimizer state dictionary, and best model parameters.
        """
        if (
            self.best_model_state_dict is not None
            and self.best_optimizer_state_dict is not None
        ):
            best_model_state_dict = model.load_state_dict(self.best_model_state_dict)
            best_optimizer_state_dict = optimizer.load_state_dict(
                self.best_optimizer_state_dict
            )
            best_model_params = self.best_model_params
            return best_model_state_dict, best_optimizer_state_dict, best_model_params


early_stopping = EarlyStopping()
