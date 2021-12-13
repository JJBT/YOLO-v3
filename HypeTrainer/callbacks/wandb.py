import wandb
import os

from .base_callback import BaseCallback


__all__ = ['WandBCallback']


class WandBCallback(BaseCallback):
    def __init__(self, frequency, **kwargs):
        super().__init__(frequency, before=True, after=False)

    def before_run(self, trainer):
        wandb.init(
            project=trainer.cfg['project_name'],
            name=trainer.cfg['run_name'],
            dir=trainer.cfg['output_path'],
            # mode='offline'
        )
        trainer.cfg['output_path'] = wandb.run.dir
        wandb.config = trainer.cfg
        wandb.watch(trainer.model, criterion=trainer.criterion, log='all', log_freq=5000, log_graph=True)

    def add_validation_metrics(self, trainer):
        wandb.log(trainer.state.validation_metrics, step=trainer.state.step)

    def add_video(self, trainer, path_to_video, caption=None):
        video = wandb.Video(path_to_video, caption=caption, fps=trainer.cfg['fps'])
        wandb.log({f'video_{caption}': video}, step=trainer.state.step)

    def __call__(self, trainer):
        wandb.log(trainer.state.last_train_loss, step=trainer.state.step)
        wandb.log({'lr': trainer.optimizer.param_groups[0]['lr']}, step=trainer.state.step)
