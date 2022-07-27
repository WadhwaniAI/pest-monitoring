"""
Create a nn.Module class that takes two jit checkpoints as input
and uses their forward passes as an output
"""
from typing import Tuple

import torch
from torch import nn

from utils import Decoder, DefaultBoxes


class CombinedModel(nn.Module):
    """torch nn module that combines and processes the validation and counting jit model outputs."""

    def __init__(
        self,
        validation_ckpt: str,
        counting_ckpt: str,
        nms_th: float,
        max_num: int,
        input_size: int,
        conf_th: float,
        boxdata: dict,
    ):
        """Load jit checkpoints, initialize decoder and generate default boxes.

        Parameters
        ----------
        validation_ckpt : str
            Path to the validation model jit checkpoint.
        counting_ckpt : str
            Path to the counting model jit checkpoint.
        nms_th : float
            NMS Threshold to be used in decoding counting model output.
        max_num : int
            Max number of boxes allowed from the counting model output.
        input_size : int
            Square dimension of the input image the models expect.
        conf_th : float
            Minimum confidence value for a box to be considered foreground.
        boxdata : dict
            Information required to generate default boxes.
        """
        super().__init__()
        self.validation_net = torch.jit.load(validation_ckpt)
        self.counting_net = torch.jit.load(counting_ckpt)
        self.nms_th = torch.tensor(nms_th)
        self.max_num = torch.tensor(max_num)
        self.input_size = torch.tensor(input_size)
        self.conf_th = torch.tensor(conf_th)
        dboxes = DefaultBoxes(input_size, **boxdata)
        self.dboxes = dboxes(order="ltrb")
        dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        nboxes = torch.tensor(self.dboxes.size(0))
        scale_xy = torch.tensor(dboxes.scale_xy)
        scale_wh = torch.tensor(dboxes.scale_wh)

        self.box_decoder = torch.jit.script(
            Decoder(self.dboxes, dboxes_xywh, nboxes, scale_xy, scale_wh)
        )
        # if you want to apply transforms

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on validation and counting models, followed by post-processing of the
        counting output.

        Parameters
        ----------
        img : torch.Tensor
            Batch with a single transformed image.

        Returns
        -------
        pvals : torch.Tensor
            Softmax of (trap, non-trap) prediciton. Unaltered output of the validation model.
        pd_abw_boxes : torch.Tensor
            Coordinates of boxes predicted as abw. Bounding box coordinates are in [x1, y1, x2, y2]
            relative (to the width and height of the image) format. Where (x1,y1) is the top-left
            corner of the box and (x2,y2) is the bottom-right corner of the box respectively.
        pd_pbw_boxes : torch.Tensor
            Coordinates of boxes predicted as pbw. Bounding box coordinates are in [x1, y1, x2, y2]
            relative (to the width and height of the image) format. Where (x1,y1) is the top-left
            corner of the box and (x2,y2) is the bottom-right corner of the box respectively.
        pd_abw_scores : torch.Tensor
            Confidence scores of the boxes in ``pd_abw_boxes``. The confidence score of box at
            ``pd_abw_boxes[i]`` will be ``pd_abw_scores[i]``.
        pd_pbw_scores : torch.Tensor
            Confidence scores of the boxes in ``pd_pbw_boxes``. The confidence score of box at
            ``pd_pbw_boxes[i]`` will be ``pd_pbw_scores[i]``.
        """
        # Convert to float
        img = img.float()

        # Make Predictions
        with torch.no_grad():
            plocs, plabels = self.counting_net(img)
            pvals = self.validation_net(img)

        # Run NMS compression on plocs and plabels
        locs, labels, scores = self.box_decoder(
            plocs, plabels, self.nms_th, self.max_num, self.conf_th
        )[0]
        locs, labels, scores = locs.detach().cpu(), labels.detach().cpu(), scores.detach().cpu()

        #         locs = self._convert_xyx2y2_to_bltr(locs)
        labels -= 1

        abw_mask = (labels == 0).nonzero(as_tuple=True)
        pbw_mask = (labels == 1).nonzero(as_tuple=True)
        pd_abw_boxes = locs[abw_mask]
        pd_pbw_boxes = locs[pbw_mask]
        pd_abw_scores = scores[abw_mask]
        pd_pbw_scores = scores[pbw_mask]

        return pvals, pd_abw_boxes, pd_pbw_boxes, pd_abw_scores, pd_pbw_scores

    def _convert_xyx2y2_to_bltr(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert from xyx2y2 relative coordinates to absolute bltr coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Box coordinates in [x1, y1, x2, y2] relative (to the width and height of the image)
            format. Where (x1,y1) is the top-left corner of the box and (x2,y2) are the width and
            height of the box respectively.

        Returns
        -------
        out : torch.Tensor
            Box coordinates in [x3, y3, x4, y4] absolute pixel format. Where (x3,y3) is the
            bottom-left corner of the box and (x4,y4) is the top-right corner of the box.
        """
        # Edit here to get bottom left top right
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        w, h = self.input_size, self.input_size
        _x, _y = (x1 * w).type(torch.LongTensor), (y1 * h).type(torch.LongTensor)
        _x2, _y2 = (x2 * w).type(torch.LongTensor), (y2 * h).type(torch.LongTensor)

        # Preventin zero size boxes
        _x2[_x2 == _x] += 1
        _y2[_y2 == _y] += 1

        _w, _h = _x2 - _x, _y2 - _y

        x0 = _x
        y0 = _y + _h

        x1 = _x + _w
        y1 = _y

        return torch.stack([x0, y0, x1, y1], axis=1)
