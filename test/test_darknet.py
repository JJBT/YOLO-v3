from typing import List
import unittest

import torch
import albumentations
from albumentations.pytorch import ToTensorV2

from dataset import CocoDataset
from darknet import Darknet53


class TestDarknet(unittest.TestCase):
    @staticmethod
    def check_output_shape(output: List[torch.Tensor]):
        assert output[0].shape == torch.Size([1, 256, 52, 52])
        assert output[1].shape == torch.Size([1, 512, 26, 26])
        assert output[2].shape == torch.Size([1, 1024, 13, 13])

    def test_darknet_0(self):
        model = Darknet53()
        test_input = torch.rand(1, 3, 416, 416)
        output = model(test_input)
        self.check_output_shape(output)

    def test_darknet_1(self):
        model = Darknet53()

        dataset = CocoDataset(
            path_to_annotations='/data/coco/annotations/instances_val2017.json',
            path_to_data='/data/coco/images/val2017',
            transforms=
            albumentations.Compose(
                [albumentations.Resize(416, 416), ToTensorV2()],
                bbox_params=albumentations.BboxParams('coco', label_fields=['labels']),

            )
        )
        test_input = dataset[0][0].float()
        output = model(test_input.unsqueeze(0))
        self.check_output_shape(output)
