from __future__ import annotations  # Remove once we move to 3.10

import functools as ft

import torch.nn as nn

from . import (
    BatchInfo,
    LossValue,
    LossTrail,
    LossAggregator,
)

#
#
#
class AbstractLoss(nn.Module):
    """Abstract class from which loss calculators should extend"""

    def __str__(self):
        """

        Returns
        -------
        str
            Name of class
        """

        return type(self).__name__

    def forward(self, pr: BatchInfo, gt: BatchInfo):
        raise NotImplementedError()


class AbstractLossCalculator(AbstractLoss):
    def __init__(self, criterion, aggregator=None):
        """

        Parameters
        ----------
        criterion: two-argument callable
            Generally used for underlying loss calculator in Pytorch
        aggregator: LossAggregator, optional
            Class to handle loss aggregation. See `LossAggregator::factory`
            for the default value
        """

        super().__init__()
        self.criterion = criterion
        self.aggregate = LossAggregator.factory(aggregator)

    def forward(self, pr: BatchInfo, gt: BatchInfo):
        """

        Parameters
        ----------
        pr: BatchInfo
            Predicted information
        gt: BatchInfo
            Ground truth information

        Returns
        -------
        LossValue
            Class name as "name" parameter, `forward` result as "loss" value
        """

        loss = LossValue(str(self), self._forward(pr, gt))
        return self.aggregate(pr, gt, loss)

    def _forward(self, pr: BatchInfo, gt: BatchInfo):
        raise NotImplementedError()


class CompositeLoss(AbstractLoss):
    """Concrete class to compose loss calculators"""

    def __init__(
        self, module: AbstractLossCalculator, weight: float = 1.0, follow: CompositeLoss = None
    ):
        """

        Parameters
        ----------
        module: AbstractLossCalculator
            An instance of a concrete loss calculator
        weight: float
            Weight that should be assigned the loss calculators loss value
        follow: CompositeLoss, optional
            Previous instance of composite loss from which this follows
        """

        super().__init__()

        self.module = module
        self.weight = weight
        self.follow = follow

    def forward(self, pr: BatchInfo, gt: BatchInfo):
        """

        Parameters
        ----------
        pr: BatchInfo
            Predicted information
        gt: BatchInfo
            Ground truth information

        Returns
        -------
        LossTrail
            Collection of LossValue's for each calculator
        """

        trail = LossTrail()
        value = self.update(self.module(pr, gt), weight=self.weight)

        trail += value
        if self.follow is not None:
            trail += self.follow(pr, gt)

        return trail

    @ft.singledispatchmethod
    def update(self, loss, **kwargs):
        """
        Generic interface for updating a loss value. This method is
        abstract: registered methods perform the update depending on
        the loss type. This family of the methods exist because loss
        calculation can return one of two types (see dtypes.py). This
        a convenience method that uses singledispatch to alleviate
        mucking with isinstance.

        Parameters
        ----------
        loss: [LossValue,LossTrail]
              Loss type that requires an update
        **kwargs:
              Key-value pairs that contain the update
        """

        raise TypeError("Loss type {} not supported}".format(type(loss)))

    @update.register
    def _(self, loss: LossValue, **kwargs):
        """
        Update a LossValue. Since the LossValue is a namedtuple,
        that essentially means calling the _replace method

        Parameters
        ----------
        loss: LossValue
              Loss value that requires the update
        **kwargs:
              Key-value pairs that correspond to attributes in LossValue
        """

        return loss._replace(**kwargs)

    @update.register
    def _(self, loss: LossTrail, **kwargs):
        """
        Update a LossTrail. This does nothing since the weight has
        already been added to the LossValue.

        Parameters
        ----------
        loss: LossTrail
              Loss trail that requires the update
        **kwargs:
              Ignored
        """

        return loss
