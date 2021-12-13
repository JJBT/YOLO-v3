import torch
import random
import numpy as np
import pydoc
from omegaconf import DictConfig
from collections import MutableMapping


def object_from_dict(d, parent=None, ignore_keys=None, **default_kwargs):
    assert isinstance(d, (dict, DictConfig)) and 'type' in d
    kwargs = d.copy()
    kwargs = dict(kwargs)
    object_type = kwargs.pop('type')

    if object_type is None:
        return None

    if ignore_keys:
        for key in ignore_keys:
            kwargs.pop(key, None)

    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    # support nested constructions
    for key, value in kwargs.items():
        if isinstance(value, (dict, DictConfig)) and 'type' in value:
            value = object_from_dict(value)
            kwargs[key] = value

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    else:
        return pydoc.locate(object_type)(**kwargs)


def freeze_layers(model, layers_to_train):
    """Freeze layers not included in layers_to_train"""
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)


def set_determenistic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, '', sep=sep).items())
        else:
            value = v.item() if isinstance(v, torch.Tensor) else v
            items.append((new_key, value))

    return dict(items)


def loss_to_dict(loss):
    if not isinstance(loss, dict):
        return {'loss': loss}
    else:
        return loss


def get_state_dict(model):
    if model is None:
        return None
    else:
        return model.state_dict()


def load_state_dict(model, state_dict):
    if model is None:
        return None
    else:
        return model.load_state_dict(state_dict)
