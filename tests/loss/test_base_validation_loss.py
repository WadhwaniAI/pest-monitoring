import unittest

import torch

from src.loss import BasicValidationLoss, BatchInfo


class BaseValidationLossTestCase(unittest.TestCase):
    """Class to run tests on BaseValidationLoss"""

    @classmethod
    def setUpClass(cls):
        cls.criterion = BasicValidationLoss()

    def test_zero_loss(self):
        pred_vclass = torch.Tensor([[1e3, 1e-6, 1e-6], [1e3, 1e-6, 1e-6], [1e-6, 1e-6, 1e3]])
        true_vclass = torch.argmax(pred_vclass, axis=1)

        pr = BatchInfo(labels=pred_vclass)
        gt = BatchInfo(labels=true_vclass)
        value = self.criterion(pr, gt)

        self.assertEqual(value.loss, 0)


if __name__ == "__main__":
    unittest.main()
