from .base_callback import BaseCallback

__all__ = ['ValidationCallback']


class ValidationCallback(BaseCallback):
    def __init__(self, frequency, **kwargs):
        super().__init__(frequency=frequency, before=False, after=False)

    def __call__(self, trainer):
        self.computed_metrics = trainer.evaluate(dataloader=trainer.val_dataloader, metrics=trainer.metrics)
        for metric_name, metric_value in self.computed_metrics.items():
            trainer.state.add_validation_metric(name=f'val/{metric_name}', value=metric_value)

        trainer.state.log_validation()
        self.log_validation_wandb(trainer)

    def log_validation_wandb(self, trainer):
        name = 'WandBCallback'
        if name in trainer.callbacks:
            wb_callback = trainer.callbacks[name]
            wb_callback.add_validation_metrics(trainer)
