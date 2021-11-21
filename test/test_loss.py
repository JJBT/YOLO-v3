import unittest
from copy import deepcopy

import albumentations
import torch
from torch.nn.functional import one_hot

from dataset import CocoDataset
from loss import YoloLoss

unittest.TestLoader.sortTestMethodsUsing = None


class TestLoss(unittest.TestCase):
    def test_loss_1(self):
        dataset = CocoDataset(
            path_to_annotations='/home/vladimir/dev/data/coco/annotations/instances_val2017.json',
            path_to_data='/home/vladimir/dev/data/coco/images/val2017',
            transforms=
            albumentations.Compose(
                [albumentations.Resize(416, 416)],
                bbox_params=albumentations.BboxParams('coco', label_fields=['labels']),

            )
        )
        loss = YoloLoss()
        image, targets = dataset[0]
        predictions = deepcopy(targets)
        for i, prediction in enumerate(predictions):
            prediction[..., 0] = torch.logit(prediction[..., 0], eps=1e-12)
            prediction[..., 1:3] = torch.logit(prediction[..., 1:3], eps=1e-12)
            prediction[..., 3:5] = torch.log(prediction[..., 3:5])
            class_ = one_hot(targets[i][..., 5].long(), num_classes=90).float()
            class_ *= 110
            class_ -= 20
            predictions[i] = torch.concat([prediction[..., :5], class_],
                                          dim=-1)

        loss_value = loss(targets, predictions)

        self.assert_(torch.isclose(loss_value['loss'], torch.tensor(0.)).bool())

    def test_loss_2(self):
        loss = YoloLoss()

        target = [torch.rand(3, 13, 13, 6), torch.rand(3, 26, 26, 6), torch.rand(3, 52, 52, 6)]
        prediction = [torch.rand(3, 13, 13, 95), torch.rand(3, 26, 26, 95), torch.rand(3, 52, 52, 95)]

        loss(target, prediction)
