import unittest

import torch
import torch.nn as nn

from src.loss import BatchInfo, DetectionSmoothL1Loss


class DetectionSmoothL1LossTestCase(unittest.TestCase):
    """Class to run tests on DetectionSmoothL1Loss"""

    @classmethod
    def setUpClass(cls):
        cls.criterion = DetectionSmoothL1Loss()

    def test_zero_loss(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_loc = true_loc_vec.clone()
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.randint(low=0, high=2, size=(N, num_dboxes))

        pr = BatchInfo(pred_loc, pred_bclass)
        gt = BatchInfo(true_loc_vec, true_bclass)
        value = self.criterion(pr, gt)

        self.assertEqual(value.loss.sum(), 0)

    def test_equal_to_smooth_l1_loss(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.ones(size=(N, num_dboxes))

        pr = BatchInfo(pred_loc, pred_bclass)
        gt = BatchInfo(true_loc_vec, true_bclass)
        detection_smooth_l1_loss = self.criterion(pr, gt)

        smooth_l1 = nn.SmoothL1Loss(reduction="none")
        smooth_l1_loss = smooth_l1(pred_loc, true_loc_vec).sum(dim=1).sum(dim=1)

        self.assertTrue(
            torch.equal(detection_smooth_l1_loss.loss, smooth_l1_loss),
        )

    def test_mask_false_zero_loss(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.zeros(size=(N, num_dboxes))

        pr = BatchInfo(pred_loc, pred_bclass)
        gt = BatchInfo(true_loc_vec, true_bclass)
        detection_smooth_l1_loss = self.criterion(pr, gt)

        self.assertEqual(detection_smooth_l1_loss.loss.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
