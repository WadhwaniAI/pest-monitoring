"""
"""

from __future__ import annotations  # Remove once we move to 3.10

from . import BatchInfo, LossValue


class LossAggregator:
    def __call__(self, pr: BatchInfo, gt: BatchInfo, loss: LossValue):
        raise NotImplementedError()

    @staticmethod
    def factory(request: LossAggregator):
        return NoAggregationAggregator() if request is None else request


class NoAggregationAggregator(LossAggregator):
    def __call__(self, pr: BatchInfo, gt: BatchInfo, loss: LossValue):
        return loss


class DetectionLossAggregator(LossAggregator):
    def __init__(self, lower=1e-6):
        self.lower = lower

    def __call__(self, pr: BatchInfo, gt: BatchInfo, loss: LossValue):
        pos_num = gt.classes.gt(0).sum(dim=1)
        value = loss.loss.mul(pos_num.gt(0)).div(pos_num.float().clamp(min=self.lower)).mean(dim=0)

        return loss._replace(loss=value)
