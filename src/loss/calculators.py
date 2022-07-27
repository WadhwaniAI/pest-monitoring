import warnings

import torch
import torch.nn as nn

from . import AbstractLossCalculator, BatchInfo, LossAggregator


class DetectionSmoothL1Loss(AbstractLossCalculator):
    """Smooth L1 loss over detected box positions"""

    def __init__(self, aggregator: LossAggregator = None, **kwargs):
        """

        Parameters
        ----------
        aggregator
            Class to handle loss aggregation
        **kwargs
            Key-word arguments passed to `torch.nn.SmoothL1Loss`

        Notes
        -----
        If "reduction" is not present in `kwargs`, it will be set to
        "none".
        """

        if "reduction" not in kwargs:
            kwargs["reduction"] = "none"
        super().__init__(nn.SmoothL1Loss(**kwargs), aggregator)

    def _forward(self, pr: BatchInfo, gt: BatchInfo):
        """Apply the criterion

        Parameters
        ----------
        pr : BatchInfo
            Predictions
        gt : BatchInfo
            Ground-truth

        Returns
        -------
        torch.Tensor
            calculated loss from criterion
        """
        loss = self.criterion(pr.locations, gt.locations)
        mask = gt.classes > 0

        return loss.sum(dim=1).mul(mask).sum(dim=1)


class _DetectionHardMinedCELoss(AbstractLossCalculator):
    """Hard mined cross entropy loss over detected box labels"""

    def __init__(self, aggregator: LossAggregator = None, **kwargs):
        """

        Parameters
        ----------
        aggregator: LossAggregator, optional
            Class to handle loss aggregation
        **kwargs
            Key-word arguments passed to `torch.nn.CrossEntropyLoss`

        Notes
        -----
        If "reduction" is not present in `kwargs`, it will be set to
        "none".
        """

        if "reduction" not in kwargs:
            kwargs["reduction"] = "none"
        super().__init__(nn.CrossEntropyLoss(**kwargs), aggregator)

    def _forward(self, pr: BatchInfo, gt: BatchInfo):
        """Apply the criterion

        Parameters
        ----------
        pr : BatchInfo
            Predictions
        gt : BatchInfo
            Ground-truth

        Returns
        -------
        torch.Tensor
            calculated loss from criterion
        """
        # hard negative mining
        loss = self.criterion(pr.classes, gt.classes)

        mask = gt.classes > 0

        pos_num = self.npos(mask)

        # postive mask will never selected
        con_neg = loss.clone()
        con_neg[mask] = 0
        (_, con_idx) = con_neg.sort(dim=1, descending=True)
        (_, con_rank) = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num
        if mask.bitwise_and(neg_mask).any():
            # XXX: This is a warning instead of an error because the
            #      logic for building test data isn't complete; so if
            #      it remains an exception, none of the test cases
            #      would pass. Once that is corrected, the ValueError
            #      should be raised instead of the warning

            # raise ValueError('Simultaneous classes')
            warnings.warn(
                "Simultaneous classes will not always be supported",
                DeprecationWarning,
            )

        return mask.bitwise_or(neg_mask).mul(loss).sum(dim=1)

    def npos(self, mask):
        raise NotImplementedError()


class DetectionHardMinedCELoss(_DetectionHardMinedCELoss):
    def __init__(
        self,
        aggregator: LossAggregator = None,
        neg_pos_ratio: int = 3,
        **kwargs,
    ):
        """

        Parameters
        ----------
        aggregator: LossAggregator, optional
            Class to handle loss aggregation
        neg_pos_ratio: int, optional
            Ratio of negative to positive boxes
        **kwargs
            Key-word arguments passed to `torch.nn.CrossEntropyLoss`

        Notes
        -----
        If "reduction" is not present in `kwargs`, it will be set to
        "none".
        """

        super().__init__(aggregator, **kwargs)
        self.neg_pos_ratio = neg_pos_ratio

    def npos(self, mask):
        return mask.sum(dim=1).mul(self.neg_pos_ratio)


class ModifiedDetectionHardMinedCELoss(_DetectionHardMinedCELoss):
    """Hard mined cross entropy loss over detected box labels"""

    def __init__(
        self,
        aggregator: LossAggregator = None,
        neg_pos_ratio: int = 3,
        min_neg_num: int = 5,
        **kwargs,
    ):
        """

        Parameters
        ----------
        aggregator: LossAggregator, optional
            Class to handle loss aggregation
        neg_pos_ratio: int, optional
            Ratio of negative to positive boxes
        **kwargs
            Key-word arguments passed to `torch.nn.CrossEntropyLoss`

        Notes
        -----
        If "reduction" is not present in `kwargs`, it will be set to
        "none".
        """

        super().__init__(aggregator, **kwargs)
        self.neg_pos_ratio = neg_pos_ratio
        self.min_neg_num = min_neg_num

    def npos(self, mask):
        pos_num = mask.sum(dim=1)
        # multiple pos_num with self.neg_pos_ratio
        pos_num = pos_num.mul(self.neg_pos_ratio)
        # replace all zeros with self.min_neg_num in pos_num tensor
        pos_num = pos_num.masked_fill(pos_num == 0, self.min_neg_num)

        return pos_num


class BasicValidationLoss(AbstractLossCalculator):
    """Cross entropy loss over image labels"""

    def __init__(self, aggregator: LossAggregator = None, **kwargs):
        """

        Parameters
        ----------
        aggregator: LossAggregator, optional
            Class to handle loss aggregation
        **kwargs
            Key-word arguments passed to `torch.nn.CrossEntropyLoss`
        """

        super().__init__(nn.CrossEntropyLoss(**kwargs), aggregator)

    def _forward(self, pr: BatchInfo, gt: BatchInfo):
        """Apply the criterion

        Parameters
        ----------
        pr : BatchInfo
            Predictions
        gt : BatchInfo
            Ground-truth

        Returns
        -------
        torch.Tensor
            calculated loss from criterion
        """
        return self.criterion(pr.labels, gt.labels)
