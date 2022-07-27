import collections as cl
import itertools as it
import unittest

import torch

from src.loss import (
    BasicValidationLoss,
    BatchInfo,
    CompositeLoss,
    DefaultSystemLoss,
    DetectionHardMinedCELoss,
    DetectionSmoothL1Loss,
    RelativeBoxTransformer,
    StandardDetectionLoss,
)
from src.utils.prior_boxes import generate_dboxes


class BaseLossWeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dboxes = generate_dboxes(model="ssd")
        cls.transformer = RelativeBoxTransformer(dboxes)
        (cls.pr, cls.gt) = (None, None)

    def setUp(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.randint(low=0, high=num_classes, size=(N, num_dboxes))
        pred_vclass = torch.rand(size=(N, num_classes))
        true_vclass = torch.randint(low=0, high=num_classes, size=(N,))

        self.pr = BatchInfo(pred_loc, pred_bclass, pred_vclass)
        self.gt = BatchInfo(true_loc, true_bclass, true_vclass)

    def test_mixed_loss_weights(self):
        upper = 3
        points = range(upper)
        keys = (
            "detection",
            "validation",
        )

        for i in it.product(points, reversed(points)):
            kwargs = {x + "_weight": y for (x, y) in zip(keys, i)}
            with self.subTest(**kwargs):
                criterion = DefaultSystemLoss(
                    transformer=self.transformer,
                    **kwargs,
                )
                loss = criterion(self.pr, self.gt)
                expected = sum(x.loss * x.weight for x in loss)
                self.assertEqual(loss.loss, expected)


if __name__ == "__main__":
    unittest.main()
