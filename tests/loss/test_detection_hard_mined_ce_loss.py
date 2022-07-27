import unittest

import numpy as np
import torch
import torch.nn as nn

from src.loss import BatchInfo, DetectionHardMinedCELoss


class DetectionHardMinedCELossTestCase(unittest.TestCase):
    """Class to run tests on DetectionHardMinedCELoss"""

    def test_zero_loss(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_loc = torch.rand(size=(N, 4, num_dboxes))

        pred_bclass = torch.empty(size=(N, num_classes, num_dboxes))
        cls_dist = [1e3]
        cls_dist.extend([1e-6] * (num_classes - 1))
        for i in range(pred_bclass.shape[0]):
            for k in range(pred_bclass.shape[2]):
                pred_bclass[i, :, k] = torch.Tensor(np.random.permutation(cls_dist))

        true_bclass = torch.argmax(pred_bclass, axis=1)

        criterion = DetectionHardMinedCELoss()
        pr = BatchInfo(pred_loc, pred_bclass)
        gt = BatchInfo(true_loc_vec, true_bclass)
        value = criterion(pr, gt)

        self.assertEqual(value.loss.sum(), 0.0)

    def test_equal_to_ce_loss_for_pos_class(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.randint(low=0, high=num_classes, size=(N, num_dboxes))

        detection_hard_mined_ce_criterion = DetectionHardMinedCELoss(
            neg_pos_ratio=0,
        )
        detection_hard_mined_ce_loss = detection_hard_mined_ce_criterion(
            BatchInfo(pred_loc, pred_bclass),
            BatchInfo(true_loc_vec, true_bclass),
        )

        pos_mask = true_bclass > 0
        ce_criterion = nn.CrossEntropyLoss(reduction="none")
        ce_loss = ce_criterion(pred_bclass, true_bclass)
        pos_loss = (pos_mask.float() * ce_loss).sum(dim=1)

        self.assertTrue(
            torch.equal(detection_hard_mined_ce_loss.loss, pos_loss),
        )

    def test_equal_to_double_of_ce_loss_for_pos_class(self):
        N = 2
        num_dboxes = 5
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = [
            [(10, 80), (10, 50), (10, 50), (50, 10), (80, 10)],
            [(10, 50), (10, 50), (10, 20), (50, 10), (50, 10)],
        ]
        pred_bclass = torch.Tensor(pred_bclass).permute(0, 2, 1)
        true_bclass = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
        true_bclass = torch.tensor(true_bclass, dtype=torch.long)

        detection_hard_mined_ce_criterion = DetectionHardMinedCELoss(
            neg_pos_ratio=1,
        )
        detection_hard_mined_ce_loss = detection_hard_mined_ce_criterion(
            BatchInfo(pred_loc, pred_bclass),
            BatchInfo(true_loc_vec, true_bclass),
        )

        pos_mask = true_bclass > 0
        ce_criterion = nn.CrossEntropyLoss(reduction="none")
        ce_loss = ce_criterion(pred_bclass, true_bclass)
        pos_loss = (pos_mask.float() * ce_loss).sum(dim=1)

        self.assertTrue(
            torch.equal(detection_hard_mined_ce_loss.loss, 2 * pos_loss),
        )

    def test_mask_false_zero_loss(self):
        N = 4
        num_classes = 10
        num_dboxes = 8732
        pred_loc = torch.rand(size=(N, 4, num_dboxes))
        true_loc_vec = torch.rand(size=(N, 4, num_dboxes))
        pred_bclass = torch.rand(size=(N, num_classes, num_dboxes))
        true_bclass = torch.zeros(size=(N, num_dboxes), dtype=torch.long)

        criterion = DetectionHardMinedCELoss()
        detection_hard_mined_ce_loss = criterion(
            BatchInfo(pred_loc, pred_bclass),
            BatchInfo(true_loc_vec, true_bclass),
        )

        self.assertEqual(detection_hard_mined_ce_loss.loss.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
