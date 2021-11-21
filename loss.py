from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from constants import ANCHORS


class YoloLoss(nn.Module):
    def __init__(
            self,
            lambda_obj: float or int = 1.0,
            lambda_noobj: float or int = 1.0,
            lambda_bbox: float or int = 1.0,
            lambda_class: float or int = 1.0,
            anchors: List[List[float]] or None = None
    ):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class

        self.anchors = anchors if anchors is not None else ANCHORS
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, targets: List[torch.Tensor], predictions: List[torch.Tensor]) -> dict:
        noobj_loss = torch.tensor(0, dtype=torch.float)
        obj_loss = torch.tensor(0, dtype=torch.float)
        bbox_loss = torch.tensor(0, dtype=torch.float)
        class_loss = torch.tensor(0, dtype=torch.float)

        for scale_idx in range(len(targets)):
            prediction = predictions[scale_idx]
            target = targets[scale_idx]

            obj_mask = target[..., 0] == 1
            noobj_mask = target[..., 0] == 0

            # no object loss
            noobj_loss += self.bce(prediction[..., 0][noobj_mask], target[..., 0][noobj_mask])

            # object loss

            # bbox loss
            prediction[..., 1:3] = torch.sigmoid(prediction[..., 1:3])  # x, y
            prediction[..., 3:5] = torch.exp(prediction[..., 3:5])  # w, h
            bbox_loss += self.mse(prediction[..., 1:5][obj_mask], target[..., 1:5][obj_mask])

            # class loss
            class_target = F.one_hot(target[..., 5].long(), num_classes=prediction.shape[-1] - 5).float()
            class_loss += self.bce(prediction[..., 5:][obj_mask], class_target[obj_mask])

        loss = self.lambda_noobj * noobj_loss + \
            self.lambda_obj * obj_loss + \
            self.lambda_bbox * bbox_loss + \
            self.lambda_class * class_loss

        return {
            'loss': loss,
            'noobj_loss': noobj_loss.detach(),
            'obj_loss': obj_loss.detach(),
            'bbox_loss': bbox_loss.detach(),
            'class_loss': class_loss.detach()
        }

