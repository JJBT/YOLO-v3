from typing import Tuple, Dict, List
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.ops import box_iou
import albumentations

from utils import read_image
from constants import ANCHORS


class CocoDataset(Dataset):
    def __init__(
            self,
            path_to_annotations: str,
            path_to_data: str,
            num_classes: int = 90,
            anchors: List[List[float]] or None = None,
            image_size: int = 416,
            grid_size: List[int] or None = None,
            transforms: albumentations.Compose or None = None,

    ):
        self.path_to_annotations = path_to_annotations
        self.path_to_data = path_to_data
        self.num_classes = num_classes
        self.image_size = image_size
        self.anchors = anchors if anchors else ANCHORS
        self.anchors = torch.tensor(self.anchors)
        self.num_anchors = self.anchors.shape[1]

        self.grid_size = grid_size if grid_size else [13, 26, 52]
        self.cell_size = [self.image_size / grid_size for grid_size in self.grid_size]  # in pixels
        self.transforms = transforms
        self.coco = COCO(self.path_to_annotations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray or torch.Tensor, List[torch.Tensor]]:
        """
        image [3x416x416] target
        :return: List[Tensor[13*13*"len(anchors[0])"*num_classes+5], Tensor[26*26*"len(anchors[1])"*num_classes+5]],
        Tensor[52*52*"len(anchors[0])"*num_classes+5]
        """
        image_idx = int(np.random.choice(self.coco.getImgIds()))
        image_info = self.coco.loadImgs(image_idx)[0]
        path_to_image = os.path.join(self.path_to_data, image_info['file_name'])
        image = read_image(path_to_image)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_idx))
        if not anns:
            return self.__getitem__(idx)

        bboxes = []
        labels = []

        for ann in anns:
            bboxes.append(ann['bbox'])  # [x_raw, y_raw, w_raw, h_raw]
            labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        bboxes = torch.tensor(bboxes).float()

        bboxes[..., 0] += bboxes[..., 2] // 2
        bboxes[..., 1] += bboxes[..., 3] // 2
        bboxes[..., 2] /= image.shape[1]
        bboxes[..., 3] /= image.shape[0]

        flattened_anchors = self.anchors.reshape(-1, 2).float()  # for iou calculating

        iou_bbox_anchors = box_iou(
            torch.concat([torch.zeros_like(bboxes[..., 2:4]), bboxes[..., 2:4]], dim=1),  # zeros for x, y
            torch.concat([torch.zeros_like(flattened_anchors), flattened_anchors], dim=1),  # zeros for x, y
        )
        iou_idxs = torch.argmax(iou_bbox_anchors, dim=1)

        target_list = [
            torch.zeros(self.num_anchors, grid, grid, 6).float() for grid in self.grid_size
            # 6 = 1 (objectness) + 4 (coords) + 1 (class)
        ]  # list of targets for each scale

        for i, idx in enumerate(iou_idxs):
            scale_idx = idx // self.anchors.shape[1]
            anchor_idx = idx % self.anchors.shape[1]
            cell_size = self.cell_size[scale_idx]
            bbox = bboxes[i]
            anchor = self.anchors[scale_idx, anchor_idx]

            x_bbox, y_bbox = (bbox[0] % cell_size) / cell_size, (bbox[1] % cell_size) / cell_size
            w_bbox, h_bbox = bbox[2] / anchor[0], bbox[3] / anchor[1]

            target_cell = torch.tensor([1., x_bbox, y_bbox, w_bbox, h_bbox, labels[i]])

            cell_x, cell_y = int(bbox[0] / cell_size), int(bbox[1] / cell_size)
            target_list[int(scale_idx)][int(anchor_idx), cell_y, cell_x] = target_cell

        return image, target_list

    def __len__(self) -> int:
        return len(self.coco.imgs)

