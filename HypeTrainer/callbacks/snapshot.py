import os
import logging
import warnings
from typing import List

import torch

from .base_callback import BaseCallback
from HypeTrainer.utils.utils import get_state_dict


__all__ = ['SaveSnapshotCallback', 'LoadSnapshotCallback']


logger = logging.getLogger(__name__)


class SaveSnapshotCallback(BaseCallback):
    def __init__(self, frequency, **kwargs):
        super().__init__(frequency=frequency, before=True, after=False)
        self.savedir = None
        self.snapshot_filename = 'snapshot_{}.pt'

    def before_run(self, trainer):
        self.savedir = os.path.join(trainer.cfg['output_path'], 'snapshots')

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)

    def __call__(self, trainer):
        step = str(trainer.state.step).zfill(6)
        filename = self.snapshot_filename.format(step)
        self._save_snapshot(trainer, filename)

    def _save_snapshot(self, trainer, filename):
        torch.save({
            'model_state_dict': get_state_dict(trainer.model),
            'trainer_state_dict': get_state_dict(trainer.state),
        }, os.path.join(self.savedir, filename))


class LoadSnapshotCallback(BaseCallback):
    # TODO call wandb callback
    def __init__(self, paths: List or str):
        super().__init__(frequency=0, before=True, after=False)
        self.paths = paths if not isinstance(paths, str) else [paths]
        self.paths = [
            self._search_latest_snapshot(path)
            if os.path.isdir(path)
            else path
            for path in self.paths
        ]

    def __call__(self, trainer):
        # before run
        for path in self.paths:
            self._load_snapshot(trainer, path)
            logger.info(f'Snapshot {path} loaded')

    @staticmethod
    def _load_snapshot(trainer, path_to_snapshot):
        snapshot = torch.load(path_to_snapshot, map_location=trainer.accelerator.device)

        if 'trainer_state' in snapshot:
            trainer.state.load_state_dict(snapshot['trainer_state_dict'])

        snapshot = snapshot.get('model_state_dict', snapshot)
        trainer.model.load_state_dict(snapshot)

    @staticmethod
    def _search_latest_snapshot(path_to_dir):
        # TODO: recursive search of latest snapshot
        return
