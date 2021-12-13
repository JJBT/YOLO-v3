from collections import defaultdict
import torch

from .utils.utils import loss_to_dict


class BaseMetric:
    def __init__(self, name: str, default_value=None, target_transform=None, prediction_transform=None, **kwargs):
        self.name = name.replace(' ', '_')
        self.default_value = default_value
        self.target_transform = target_transform if target_transform else lambda x: x
        self.prediction_transform = prediction_transform if prediction_transform else lambda x: x

    def prepare(self, y: torch.Tensor, y_pred: torch.Tensor):
        y = self.target_transform(y)
        y_pred = self.prediction_transform(y_pred)

        if isinstance(y, torch.Tensor):
            y = y.detach()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach()

        return y, y_pred

    def step(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class TorchLossMetric(BaseMetric):
    def __init__(self, name, loss_fn, target_transform=None, prediction_transform=None, **kwargs):
        super().__init__(name, float('inf'), target_transform, prediction_transform)
        self.loss_fn = loss_fn

        self._total = 0
        self._loss_sum_dict = defaultdict(lambda: 0)

    def step(self, y: torch.Tensor, y_pred: torch.Tensor):
        y, y_pred = self.prepare(y, y_pred)

        loss_dict = self.loss_fn(y_pred, y)
        loss_dict = loss_to_dict(loss_dict)

        for loss_name, loss_value in loss_dict.items():
            self._loss_sum_dict[loss_name] += loss_value.item()

        self._total += 1

    def compute(self):
        if self._total == 0:
            result = dict.fromkeys(self._loss_sum_dict.keys(), 0)
        else:
            result = dict()
            for loss_name, loss_value in self._loss_sum_dict.items():
                result[loss_name] = loss_value / self._total

        return result

    def reset(self):
        self._loss_sum_dict = defaultdict(lambda: 0)
        self._total = 0
