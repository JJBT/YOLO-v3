import json

__all__ = [
    'save_json',
    'load_json',
]


def save_json(obj, path_to_save):
    with open(path_to_save, 'w') as file:
        json.dump(obj, file, indent=0)


def load_json(path_to_load):
    with open(path_to_load, 'r') as file:
        obj = json.load(file)

    return obj
