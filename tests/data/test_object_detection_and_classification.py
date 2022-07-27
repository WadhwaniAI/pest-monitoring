"""Tests src.data.base_dataset.BaseDataset"""
import os
import random
import unittest

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data.transforms import Compose


class TestObjectDetectionCase(unittest.TestCase):
    """Class to run tests on src.data.base_dataset.BaseDataset for the
    Combined (Object Detection + Classication) Dataset Case"""

    @classmethod
    def setUpClass(cls):
        # Load Object Detection + Classification Dataset
        base_configs_path = "tests/helpers/resources/configs/datamodule/dataset"
        config_name = os.path.join(base_configs_path, "test-general-dataset.yaml")
        cls.args = OmegaConf.load(config_name)

    def test_len(self):
        """Tests the length of the dataset"""
        mode_len_dict = {"train": 120, "val": 40, "test": 40}
        for mode in ["train", "val", "test"]:
            with self.subTest(mode=mode):
                dataset = self._get_dataset(mode)
                self.assertEqual(
                    len(dataset), mode_len_dict[mode], f"Length of {mode} dataset is incorrect"
                )

    def test_get_item(self):
        """Tests _get_item output for the object detection dataset"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        record = dataset[idx]

        self.assertIsInstance(record, dict, "Record is not a dictionary")

        keys = ["img_id", "img", "bbox_class", "bbox_coord", "label_class", "label_value"]
        for key in keys:
            with self.subTest(key=key):
                self.assertIn(key, record, f"{key} not in record")

    def test_image(self):
        """Tests the image output of the dataset"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        img = dataset[idx]["img"]

        self.assertIsInstance(img, torch.Tensor, "Image is not a torch tensor")
        self.assertEqual(img.shape, torch.Size([3, 300, 300]), "Image shape is incorrect")

    def test_label_class(self):
        """Tests the label_class output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 0
        label_class = dataset[idx]["label_class"]

        self.assertIsNotNone(label_class, "label_class is None")
        self.assertIsInstance(label_class, torch.Tensor, "label_class is not a torch tensor")
        self.assertTrue(
            torch.allclose(label_class, torch.tensor([5, 6, 7, 8], dtype=torch.int32), atol=1e-4),
            "label_class is incorrect",
        )

    def test_label_value(self):
        """Tests the label_value output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 0
        label_value = dataset[idx]["label_value"]

        self.assertIsNotNone(label_value, "label_value is None")
        self.assertIsInstance(label_value, torch.Tensor, "label_value is not a torch tensor")
        self.assertTrue(
            torch.allclose(label_value, torch.tensor([3.0, 7.0, 85.0, 1.0]), atol=1e-4),
            "label_value is incorrect",
        )

    def test_bbox_coord(self):
        """Tests the bbox_coord output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 1
        bbox_coord = dataset[idx]["bbox_coord"]

        self.assertIsNotNone(bbox_coord, "bbox_coord is None")
        self.assertIsInstance(bbox_coord, torch.Tensor, "bbox_coord is not a torch tensor")
        self.assertEqual(bbox_coord.shape, torch.Size([4, 4]), "bbox_coord shape is incorrect")

    def test_bbox_class(self):
        """Tests the bbox_class output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 1
        bbox_class = dataset[idx]["bbox_class"]

        self.assertIsNotNone(bbox_class, "bbox_class is None")
        self.assertIsInstance(bbox_class, torch.Tensor, "bbox_class is not a torch tensor")
        self.assertEqual(bbox_class.shape, torch.Size([4]), "bbox_class shape is incorrect")

    # Helper functions
    def _get_dataset(self, mode: str):
        """Helper to get dataset for a given mode"""
        transforms = self._get_transforms(mode)
        return instantiate(
            self.args, dataset_config=self.args, mode=mode, transforms=transforms, _recursive_=False
        )

    def _get_transforms(self, mode: str):
        """Helper to get transforms for a given mode"""
        transforms = self.args.transforms
        if mode == "train":
            return Compose([hydra.utils.instantiate(transform) for transform in transforms.train])
        elif mode == "val":
            return Compose([hydra.utils.instantiate(transform) for transform in transforms.val])
        elif mode == "test":
            return Compose([hydra.utils.instantiate(transform) for transform in transforms.test])
        else:
            raise ValueError(f"Unknown mode {mode}")


if __name__ == "__main__":

    unittest.main()
