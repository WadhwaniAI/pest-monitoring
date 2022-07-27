"""Tests src.data.base_dataset.BaseDataset with basic tests"""
import os
import unittest

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.transforms import Compose


class BaseDatasetBasicTests(unittest.TestCase):
    """Tests src.data.base_dataset.BaseDataset with basic tests"""

    @classmethod
    def setUpClass(cls):
        # Load Generalized Dataset
        base_configs_path = "tests/helpers/resources/configs/datamodule/dataset"
        config_name = os.path.join(base_configs_path, "test-general-dataset.yaml")
        cls.args = OmegaConf.load(config_name)

    def test_instantiation(self):
        """Tests if the dataset has been instantiated correctly"""
        for mode in ["train", "val", "test"]:
            with self.subTest(mode=mode):
                try:
                    self._get_dataset(mode)
                except Exception as e:
                    self.fail(f"Dataset instantiation failed for mode {mode} with error {e}")

    def test_dataset_type(self):
        """Test the output of the instantiation"""
        for mode in ["train", "val", "test"]:
            with self.subTest(mode=mode):
                dataset = self._get_dataset(mode)
                self.assertIsInstance(
                    dataset, Dataset, f"{mode} dataset is not a torch.data.Dataset"
                )

    def test_len(self):
        """Tests if __len__ has been implemented correctly"""
        dataset = self._get_dataset("train")
        try:
            len(dataset)
        except Exception as e:
            self.fail(f"__len__ not implemented: {e}")

    def test_get_item(self):
        """Tests if __getitem__ has been implemented correctly"""
        dataset = self._get_dataset("train")
        try:
            dataset[0]
        except Exception as e:
            self.fail(f"__getitem__ not implemented: {e}")

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
