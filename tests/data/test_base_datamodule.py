"""Tests src.data.base_datamodule.BaseDataModule"""
import unittest

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule


class BaseDatamoduleTestCase(unittest.TestCase):
    """Class to run tests on BaseDatamodule"""

    @classmethod
    def setUpClass(cls):
        # Load BaseDatamodule using config
        config_name = "tests/helpers/resources/configs/datamodule/test-base-datamodule.yaml"
        args = OmegaConf.load(config_name)

        # Initialize the BaseDatamodule
        cls.datamodule: LightningDataModule = instantiate(args, data_config=args, _recursive_=False)
        cls.datamodule.setup()
        cls.train_dataloader = cls.datamodule.train_dataloader()
        cls.val_dataloader = cls.datamodule.val_dataloader()
        cls.batch_size = args.batch_size

    def _test_batch(self, batch):
        self.assertIsInstance(batch, dict, "Should return a dict")

        self.assertIsInstance(batch["img"], torch.Tensor, "img should be tensor type")
        for key, val in batch.items():
            self.assertEqual(len(val), self.batch_size, f"batch size for key: {key} not correct.")

    def test_collate(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(self.datamodule.dataset_train.__getitem__(i))
        batch = self.datamodule.collate_fn(batch)
        self._test_batch(batch)

    def test_dataloader(self):
        for i, batch in enumerate(self.train_dataloader):
            break
        self._test_batch(batch)

    def test_reproducibility(self):
        for i, batch in enumerate(self.val_dataloader):
            break
        batch1 = batch.copy()

        for i, batch in enumerate(self.val_dataloader):
            break
        batch2 = batch.copy()

        # Assert that the two batches are the same
        self.assertListEqual(
            batch1["img_id"], batch2["img_id"], "The two batches should be the same"
        )


if __name__ == "__main__":
    unittest.main()
