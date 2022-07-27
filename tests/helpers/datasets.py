import random

import torch
from torch.utils.data import Dataset


class TestObjectDetectionDataset(Dataset):
    """Test Pytorch Object Detection Dataset Object
    Built in order to test other components used in our codebase
    """

    def __init__(self):
        super(TestObjectDetectionDataset, self).__init__()
        self.length = 200

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = random.randint(0, 10)
        return {
            "img_id": "test_img",
            "img": torch.randn(3, 300, 300),
            "bbox_coord": torch.randn(n, 4),
            "bbox_class": torch.ones(
                n,
            ),
            "val_class": 0,
        }
