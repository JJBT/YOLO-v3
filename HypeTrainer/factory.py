import torch
from torch.utils.data import DataLoader
import os

from HypeTrainer.utils.utils import object_from_dict

from .empty import EmptyDataset


class BaseFactory:
    def __init__(self, cfg):
        self.cfg = cfg

        output_dir = self.cfg['output_path']
        os.makedirs(output_dir, exist_ok=True)

    def create_model(self):
        raise NotImplementedError()

    def create_optimizer(self, model: torch.nn.Module):
        raise NotImplementedError()

    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError()

    def create_criterion(self):
        raise NotImplementedError()

    def create_train_dataloader(self):
        raise NotImplementedError()

    def create_val_dataloader(self):
        dataset = EmptyDataset()
        val_dataloader = self.create_dataloader(1, dataset)
        return val_dataloader

    def create_dataset(self, cfg):
        raise NotImplementedError()

    def create_dataloader(self, bs, dataset):
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
        return dataloader

    def create_metrics(self):
        return []

    def create_callbacks(self):
        return []

    def create_augmentations(self, cfg):
        raise NotImplementedError()

    def create_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConfigFactory(BaseFactory):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.metadata = metadata
        self._attrs = dict()

    def create_model(self):
        path_to_pca_components = os.path.join(self.cfg['dataset']['data_root'], self.metadata['info']['components'])
        return object_from_dict(self.cfg['model'], path_to_pca_components=path_to_pca_components)

    def create_optimizer(self, model: torch.nn.Module):
        return object_from_dict(self.cfg['optimizer'], params=filter(lambda x: x.requires_grad, model.parameters()))

    def create_criterion(self):
        ref_frame = self._attrs['ref_frame']
        criterion = object_from_dict(self.cfg['loss'], ref_frame=ref_frame)
        self._attrs['loss_fn'] = criterion
        return criterion

    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        return None

    def create_callbacks(self):
        callbacks = []
        for callback_cfg in self.cfg['callbacks']:
            callbacks.append(object_from_dict(callback_cfg, metadata=self.metadata))

        return callbacks

    def create_metrics(self):
        metrics = []
        for metric_cfg in self.cfg['metrics']:
            metrics.append(object_from_dict(metric_cfg, **self._attrs))

        return metrics

    def create_train_dataloader(self):
        dataset = self.create_dataset({**self.cfg['dataset'], 'split': 'train'})
        self._attrs['ref_frame'] = dataset.get_ref_frame()[1]
        train_dataloader = self.create_dataloader(self.cfg['bs'], dataset)
        return train_dataloader

    def create_val_dataloader(self):
        dataset = self.create_dataset({**self.cfg['dataset'], 'split': 'val'})
        train_dataloader = self.create_dataloader(self.cfg['bs'], dataset)
        return train_dataloader

    def create_dataset(self, cfg):
        return object_from_dict(cfg)

    def create_augmentations(self, cfg):
        pass
