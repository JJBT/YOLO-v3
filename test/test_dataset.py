import unittest

import albumentations

from dataset import CocoDataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = CocoDataset(
            path_to_annotations='/home/vladimir/dev/data/coco/annotations/instances_val2017.json',
            path_to_data='/home/vladimir/dev/data/coco/images/val2017',
            transforms=
            albumentations.Compose(
                [albumentations.Resize(416, 416)],
                bbox_params=albumentations.BboxParams('coco', label_fields=['labels']),

            )
        )
        elem = dataset[0]
        print(elem[1][0].sum())
        print(elem[1][1].sum())
        print(elem[1][2].sum())
