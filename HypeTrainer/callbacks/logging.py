import os

from omegaconf import OmegaConf

from .base_callback import BaseCallback
from HypeTrainer.utils.json_utils import save_json


__all__ = ['LogCallback']


class LogCallback(BaseCallback):
    def __init__(self, frequency, **kwargs):
        super().__init__(frequency=frequency, before=True, after=True)

    def before_run(self, trainer):
        OmegaConf.save(trainer.cfg, os.path.join(trainer.cfg['output_path'], 'config.yaml'))

    def __call__(self, trainer):
        trainer.state.log_train()
