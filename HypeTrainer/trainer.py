import signal
from typing import List
import torch
import logging
from collections import OrderedDict
from accelerate import Accelerator

from .utils.utils import set_determenistic, flatten_dict, loss_to_dict
from .callbacks.base_callback import BaseCallback
from .callbacks.stop_criterion import StopAtStep


logger = logging.getLogger(__name__)


class State:
    def __init__(self):
        self.step = 0
        self.last_train_loss = None

        self.validation_metrics = dict()

    def get_validation_metric(self, name):
        return self.validation_metrics[name]

    def get(self, attribute_name: str):
        return getattr(self, attribute_name)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def state_dict(self):
        return self.__dict__

    def add_attribute(self, name, value):
        if not hasattr(self, name):
            setattr(self, name, value)

    def add_validation_metric(self, name, value):
        self.validation_metrics[name] = value

    def reset(self):
        self.step = 0
        self.last_train_loss = None

    def update(self, loss_dict=None):
        self.step += 1
        if loss_dict is not None:
            self.last_train_loss = flatten_dict(loss_dict)

    def log_train(self):
        msg = f'Step - {self.step} '
        for name, value in self.last_train_loss.items():
            msg += f'{name} - {value:.7f} '

        logger.info(msg)

    def log_validation(self):
        msg = f'Validation '
        for name, value in self.validation_metrics.items():
            msg += f'{name} - {value:.7f} '

        logger.info(msg)


class Trainer:
    def __init__(self, cfg, factory):
        signal.signal(signal.SIGINT, self._soft_exit)
        set_determenistic()

        self.train_dataloader = factory.create_train_dataloader()
        self.val_dataloader = factory.create_val_dataloader()
        self.state = State()
        self.criterion = factory.create_criterion()
        self.model = factory.create_model()
        self.optimizer = factory.create_optimizer(self.model)
        self.scheduler = factory.create_scheduler(self.optimizer)
        self.n_steps = cfg['n_steps']
        self.stop_condition = StopAtStep(last_step=self.n_steps)
        self.metrics = factory.create_metrics()
        self.device = factory.create_device()
        callbacks = factory.create_callbacks()
        self.callbacks = OrderedDict()
        self.register_callbacks(callbacks)

        self.cfg = cfg
        self.stop_validation = False
        self.accelerator = Accelerator(cpu=bool(self.device.type == 'cpu'))
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.train_dataloader, self.val_dataloader = \
            self.accelerator.prepare(self.train_dataloader, self.val_dataloader)

    def get_train_batch(self):
        if not getattr(self, 'train_data_iter', False):
            self.train_data_iter = iter(self.train_dataloader)
        try:
            batch = next(self.train_data_iter)
        except StopIteration:
            self.train_data_iter = iter(self.train_dataloader)
            batch = next(self.train_data_iter)

        return batch

    def run_step(self, batch):
        self.optimizer.zero_grad()

        inputs, targets = self.get_input_and_target_from_batch(batch)

        outputs = self.model(inputs)

        loss_dict = self.criterion(outputs, targets)

        loss_dict = loss_to_dict(loss_dict)
        loss = loss_dict['loss']
        self.accelerator.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        loss_dict['loss'] = loss_dict['loss'].detach()

        return loss_dict

    def run_train(self, n_steps=None):
        if n_steps is not None:
            self.stop_condition = StopAtStep(last_step=n_steps)

        self.state.reset()
        self.model.train()

        self._before_run_callbacks()

        while not self.stop_condition(self.state):
            batch = self.get_train_batch()
            loss = self.run_step(batch)
            self.state.update(loss)

            self._run_callbacks()

        self._after_run_callbacks()
        logger.info('Done')

    def evaluate(self, dataloader=None, metrics=None):
        if dataloader is None:
            dataloader = self.val_dataloader
        if metrics is None:
            metrics = self.metrics

        previous_training_flag = self.model.training

        self.model.eval()
        for metric in metrics:
            metric.reset()

        with torch.no_grad():
            for batch in dataloader:
                if self.stop_validation:
                    break

                inputs, targets = self.get_input_and_target_from_batch(batch)
                outputs = self.model(inputs)

                for metric in metrics:
                    metric.step(y=targets, y_pred=outputs)

        metrics_computed = {metric.name: metric.compute() for metric in metrics}
        self.model.train(previous_training_flag)

        return flatten_dict(metrics_computed)

    def get_input_and_target_from_batch(self, batch):
        return batch[0], batch[1]

    def register_callbacks(self, callbacks: List[BaseCallback]):
        for callback in callbacks:
            self.register_callback(callback)

    def register_callback(self, callback: BaseCallback):
        callback.set_trainer(self)
        callback_name = callback.__class__.__name__
        self.callbacks[callback_name] = callback

    def _before_run_callbacks(self):
        for name, callback in self.callbacks.items():
            callback.before_run(self)

    def _after_run_callbacks(self):
        for name, callback in self.callbacks.items():
            callback.after_run(self)

    def _run_callbacks(self):
        for name, callback in self.callbacks.items():
            freq = callback.frequency
            if freq != 0 and self.state.step % freq == 0:
                callback(self)

    def _soft_exit(self, sig, frame):
        logger.info('\nSoft exit... Currently running steps will be finished')
        self.stop_condition = lambda state: True
        self.stop_validation = True
