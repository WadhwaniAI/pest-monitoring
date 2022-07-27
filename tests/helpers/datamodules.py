import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.base_collate import dict_collate

from .datasets import TestObjectDetectionDataset


class TestObjectDetectionDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.collate_fn = dict_collate

    def setup(self):
        self.dataset = TestObjectDetectionDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
