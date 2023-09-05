from typing import Iterator
import torch
from torch import nn
from torch.optim import Optimizer


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_model_state_dict = None
        self.best_model_params = None
        self.best_optimizer_params = None
        self.min_test_loss = np.inf
        self.stop = False
        self.counter = 0

    def __call__(
        self, test_loss: float, model: nn.Module, optimizer: Optimizer
    ) -> None:
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
        self, model: nn.Module, optimizer: Optimizer
    ) -> tuple[dict, dict, Iterator]:
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
