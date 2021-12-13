import unittest
import os
import shutil

import cv2
import albumentations

from dataset import CocoDataset
from utils import draw_from_yolo_target


class TestDataset(unittest.TestCase):
    def test_dataset_0(self):
        dataset = CocoDataset(
            path_to_annotations='/data/coco/annotations/instances_val2017.json',
            path_to_data='/data/coco/images/val2017',
            transforms=
            albumentations.Compose(
                [albumentations.Resize(416, 416)],
                bbox_params=albumentations.BboxParams('coco', label_fields=['labels']),

            )
        )
        for i in range(len(dataset)):
            if i == 6:
                break
            image, target = dataset[i]
            # image = draw_from_yolo_target(image, target)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f'/home/vladimir/Desktop/out/{i}.jpg', image)

