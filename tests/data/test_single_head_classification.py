"""Tests src.data.pest_datasets.PestMultiHeadDataset"""
import os
import random
import unittest

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data.transforms import Compose


class TestSingleHeadClassificationCase(unittest.TestCase):
    """Class to run tests on src.data.pest_datasets.PestMultiHeadDataset for the
    Single Head Classification Case
    """

    @classmethod
    def setUpClass(cls):
        # Load Classification Single Head Dataset
        base_configs_path = "tests/helpers/resources/configs/datamodule/dataset"
        config_name = os.path.join(base_configs_path, "test-image-single-head-clf-dataset.yaml")
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

        keys = [
            "img_id",
            "img",
            "bbox_class",
            "bbox_coord",
            "label_value_cat",
            "label_value_reg",
            "label_class_cat",
            "label_class_reg",
        ]
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

    def test_label_class_cat(self):
        """Tests the label_class_cat output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 0
        label_class_cat = dataset[idx]["label_class_cat"]

        self.assertIsNotNone(label_class_cat, "label_class_cat is None")
        self.assertIsInstance(
            label_class_cat, torch.Tensor, "label_class_cat is not a torch tensor"
        )
        self.assertEqual(
            label_class_cat,
            torch.tensor([0], dtype=torch.int32),
            "label_class_cat is incorrect",
        )

    def test_label_value_cat(self):
        """Tests the label_value_cat output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = 0
        label_value_cat = dataset[idx]["label_value_cat"]

        self.assertIsNotNone(label_value_cat, "label_value_cat is None")
        self.assertIsInstance(
            label_value_cat, torch.Tensor, "label_value_cat is not a torch tensor"
        )
        self.assertEqual(label_value_cat, torch.tensor([0]), "label_value_cat is incorrect")

    def test_label_value_reg(self):
        """Tests the label_value_reg output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        label_value = dataset[idx]["label_value_reg"]

        # check if label_value is none
        self.assertIsNone(label_value, "label_value is not None")

    def test_label_class_reg(self):
        """Tests the label_class_reg output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        label_class = dataset[idx]["label_class_reg"]

        # check if label_class is none
        self.assertIsNone(label_class, "label_class is not None")

    def test_bbox_coord(self):
        """Tests the bbox_coord output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        bbox_coord = dataset[idx]["bbox_coord"]

        # check if bbox_coord is none
        self.assertIsNone(bbox_coord, "bbox_coord is not None")

    def test_bbox_class(self):
        """Tests the bbox_class output from dataset.__get_item__"""
        dataset = self._get_dataset("train")
        idx = random.randint(0, len(dataset) - 1)
        bbox_class = dataset[idx]["bbox_class"]

        # check if bbox_class is none
        self.assertIsNone(bbox_class, "bbox_class is not None")

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
