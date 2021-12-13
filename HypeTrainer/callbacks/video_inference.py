import os
from typing import Union
import shutil

from torch import nn
import torch

from lipsync3d.inference import VideoPipeline
from .base_callback import BaseCallback


__all__ = ['VideoInferenceCallback']


class VideoInferenceCallback(BaseCallback):
    def __init__(
            self,
            frequency: int,
            metadata: dict,
            start_frame_idx: int = 0,
            end_frame_idx: Union[int, float] = float('inf'),
            **kwargs
    ):
        super().__init__(frequency, before=True)
        self.path_to_save = None
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.metadata = metadata
        self.folders = ['vertices_image', 'vertices', 'atlases', 'frames']

    def before_run(self, trainer):
        self.path_to_save = os.path.join(trainer.cfg['output_path'], 'videos')
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=True)

    def __call__(self, trainer):
        path_to_save = os.path.join(self.path_to_save, str(int(trainer.state.step)))
        video_pipeline = VideoPipeline(
            path_to_video=os.path.join(self.metadata['info']['path'], 'frames'),
            path_to_audio=os.path.join(trainer.cfg['dataset']['data_root'], self.metadata['info']['audio']),
            path_to_save=path_to_save,
            fps=self.metadata['info']['fps'],
            device=trainer.accelerator.device
        )
        paths = video_pipeline.run(trainer.model, start_frame_idx=self.start_frame_idx, end_frame_idx=self.end_frame_idx)

        name = 'WandBCallback'
        if name in trainer.callbacks:
            wb_callback = trainer.callbacks[name]
            for caption, path in paths.items():
                wb_callback.add_video(trainer, path, caption=caption)

        self.postprocess(path_to_save)

    def postprocess(self, path):
        """delete all folders"""
        for folder in self.folders:
            shutil.rmtree(os.path.join(path, folder))
