import itertools as it
import operator as op
import unittest

import torch

from src.loss import (
    BatchInfo,
    DetectionLossAggregator,
    LossValue,
    RelativeBoxTransformer,
    StandardDetectionLoss,
)
from src.utils.prior_boxes import generate_dboxes


class BoxAggregationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aggregator = DetectionLossAggregator()
        cls.loss = None

    def setUp(self):
        N = 5
        num_dboxes = 10
        num_classes = 3
        loss = LossValue("", torch.ones(N) * num_dboxes)
        true_bclass = torch.randint(low=1, high=num_classes, size=(N, num_dboxes))
        gt = BatchInfo(classes=true_bclass)
        self.loss = self.aggregator(None, gt, loss).loss

    def test_returns_single_value(self):
        self.assertEqual(len(self.loss.shape), 0)

    def test_return_type_tensor(self):
        self.assertIsInstance(self.loss, torch.FloatTensor)

    def test_special_case_unity(self):
        self.assertEqual(self.loss, 1)


class BaseDetectionLossTestCase(unittest.TestCase):
    """Class to run tests on DetectionHardMinedCELoss"""

    def test_mixed_loss_weights(self):
        dboxes = generate_dboxes(model="ssd")
        transformer = RelativeBoxTransformer(dboxes)

        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.randint(low=0, high=num_classes, size=(N, num_dboxes))
        pr = BatchInfo(pred_loc, pred_bclass)
        gt = BatchInfo(true_loc, true_bclass)

        upper = 3
        points = range(upper)
        keys = (
            "location_weight",
            "confidence_weight",
        )
        for i in it.product(points, reversed(points)):
            kwargs = {x: y / (upper - 1) for (x, y) in zip(keys, i)}
            with self.subTest(**kwargs):
                criterion = StandardDetectionLoss(
                    transformer=transformer,
                    **kwargs,
                )
                value = criterion(pr, gt)

                # If the order of composition in StandardDetectionLoss
                # is misaligned with the values in `keys` this will
                # fail
                components = map(op.itemgetter(0), criterion.composition)
                expected = 0
                for (j, k) in zip(components, keys):
                    (val,) = value[str(j)]
                    expected += val.loss * kwargs[k]
                self.assertEqual(value.loss, expected)


if __name__ == "__main__":
    unittest.main()
