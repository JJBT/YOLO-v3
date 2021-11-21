import os

import cv2
import numpy as np
import torch
from typing import List
import cv2

from constants import ANCHORS


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def multiscale_from_yolo_target(targets: List[torch.Tensor], image_size: int) -> List[List[float]]:
    result = []
    # scale loop
    for i in range(len(targets)):
        # anchor loop
        for j in range(targets[i].shape[0]):
            result.extend(from_yolo_target(targets[i][j][..., :-1], image_size, ANCHORS[i][j]))

    return result


def from_yolo_target(target: torch.Tensor, img_size: int, anchor: List[float]) -> List[List[float]]:
    """Returns list of bboxes in ``(xl, yt, w, h)`` format"""
    if torch.is_tensor(target):
        target = target.detach().cpu()

    img_size = (img_size, img_size)
    grid_size = (target.shape[0], target.shape[1])

    cell_size = img_size[0] // grid_size[0], img_size[1] // grid_size[1]
    new_target = target.clone()
    new_target = torch.stack(torch.split(new_target, 5, dim=-1))

    x_offset = torch.repeat_interleave(torch.unsqueeze(torch.arange(grid_size[1]), dim=0), repeats=grid_size[0], dim=0) * cell_size[0]
    new_target[..., 1] = x_offset + new_target[..., 1] * cell_size[0]
    y_offset = torch.repeat_interleave(torch.unsqueeze(torch.arange(grid_size[0]), dim=0).T, repeats=grid_size[1], dim=1) * cell_size[1]
    new_target[..., 2] = y_offset + new_target[..., 2] * cell_size[1]

    new_target[..., 3] = new_target[..., 3] * anchor[0] * img_size[1]
    new_target[..., 4] = new_target[..., 4] * anchor[1] * img_size[0]

    new_target[..., 1] -= new_target[..., 3] // 2
    new_target[..., 2] -= new_target[..., 4] // 2

    new_target = new_target[new_target[..., 0] > 0]

    return new_target


def draw_from_yolo_target(image: np.ndarray, target: List[torch.Tensor]):
    bboxes = multiscale_from_yolo_target(target, image.shape[0])
    for bbox in bboxes:
        start_point = (int(bbox[1]), int(bbox[2]))
        end_point = (int(bbox[1] + bbox[3]), int(bbox[2] + bbox[4]))
        image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)

    return image
