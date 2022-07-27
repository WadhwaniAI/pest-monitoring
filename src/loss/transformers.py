"""
Encapsulation of box transformations to make loss calculation correct
"""

import torch
import torch.nn as nn

from . import BatchInfo


class BoxTransformer(nn.Module):
    """Abstract box transformation interface. Inherits from
    `nn.Module` to automate hardward targeting (whether
    instances live on the CPU or GPU).

    """

    def forward(self, x):
        """
        Definition of forward as required by `nn.Module`. However,
        because this data type is not meant to be a module, there
        is no implementation. Instead a an exception is thrown.

        Raises
        ------
        TypeError
            BoxTransformer's are not meant to be called in the
            "forward" context.

        """

        raise TypeError("Forward not supported")

    def expand(self, info: BatchInfo):
        raise NotImplementedError()


class BoxPreserver(BoxTransformer):
    """Basic box "preservation" transformer

    Does nothing to a box
    """

    def expand(self, info: BatchInfo):
        """Expand transform the boxes. Identity transform in this case.

        Parameters
        ----------
        info : BatchInfo
            Batch of boxes

        Returns
        -------
        BatchInfo
            Transformed batch of boxes. Same as input here.
        """
        return info


class RelativeBoxTransformer(BoxTransformer):
    """Relative to absolute box transformation"""

    def __init__(self, dboxes, lower=1e-6):
        super().__init__()

        self.lower = lower
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        data = dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0)
        self.dboxes = nn.Parameter(data, requires_grad=False)

    def expand(self, info: BatchInfo):
        """Expand transform the boxes.

        Parameters
        ----------
        info : BatchInfo
            batch of boxes

        Returns
        -------
        BatchInfo
            Transformed batch of boxes.
        """
        view = self.dboxes[:, 2:, :]
        diff = info.locations[:, :2, :] - self.dboxes[:, :2, :]
        normalized = info.locations[:, 2:, :] / view

        gxy = self.scale_xy * diff / view
        gwh = self.scale_wh * normalized.log()
        locations = torch.cat((gxy, gwh), dim=1).contiguous()

        return info._replace(locations=locations)
