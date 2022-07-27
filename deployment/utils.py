""" Inspired from
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py
"""

import itertools
from math import sqrt
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_iou


class Decoder(torch.nn.Module):
    """Transform between (bboxes, labels) <-> SSD output."""

    def __init__(
        self,
        dboxes: torch.Tensor,
        dboxes_xywh: torch.Tensor,
        nboxes: torch.Tensor,
        scale_xy: torch.Tensor,
        scale_wh: torch.Tensor,
    ):
        """Initialize the decoder.

        Parameters
        ----------
        dboxes : torch.Tensor
            default boxes in ltrb format
        dboxes_xywh : torch.Tensor
            default boxes in xywh format
        nboxes : torch.Tensor
            number of dboxes
        scale_xy : torch.Tensor
            scale in xy
        scale_wh : torch.Tensor
            scale in wh
        """
        super(Decoder, self).__init__()
        self.dboxes = dboxes
        self.dboxes_xywh = dboxes_xywh
        self.nboxes = nboxes
        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

    def scale_back_batch(
        self, bboxes_in: torch.Tensor, scores_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Do scale and transform from xywh to ltrb suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            unscaled bboxes
        scores_in : torch.Tensor
            unscaled scores

        Returns
        -------
        bboxes_out : torch.Tensor
            scaled bboxes
        scores_out : torch.Tensor
            scaled scores
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = (
            bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3],
            bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3],
        )

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def forward(
        self,
        bboxes_in: torch.Tensor,
        scores_in: torch.Tensor,
        nms_th: torch.Tensor,
        max_num: torch.Tensor,
        conf_th: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        """The decoder call for batch of image outputs.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            Bounding Boxes to be decoded.
        scores_in : torch.Tensor
            Confidence Scores corresponding to the bounding boxes.
        nms_th : torch.Tensor
            IOU threshold to be used in NMS.
        max_num : torch.Tensor
            Maximum number of output boxes allowed after decoding.
        conf_th : torch.Tensor
            Confidence threshold to filter low-confidence boxes.

        Returns
        -------
        List[List[torch.Tensor]]
            List of decoded [Bboxes, Labels, Scores] for every image
        """
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = torch.jit.annotate(List[List[Tensor]], [])
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_th, max_num, conf_th))
        return output

    def decode_single(
        self,
        bboxes_in: torch.Tensor,
        scores_in: torch.Tensor,
        nms_th: torch.Tensor,
        max_num: torch.Tensor,
        conf_th: torch.Tensor,
    ) -> List[torch.Tensor]:
        """The decoder call for a single image's outputs.

        Parameters
        ----------
        bboxes_in : torch.Tensor
            Bounding Boxes to be decoded.
        scores_in : torch.Tensor
            Confidence Scores corresponding to the bounding boxes.
        nms_th : torch.Tensor
            IOU threshold to be used in NMS.
        max_num : torch.Tensor
            Maximum number of output boxes allowed after decoding.
        conf_th : torch.Tensor
            Confidence threshold to filter low-confidence boxes.

        Returns
        -------
        out : List[torch.Tensor]
            Decoded [Bboxes, Labels, Scores] for the input image.
        """
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i in torch.arange(1, scores_in.size(1)):

            score = scores_in[:, i]
            mask = score.gt(conf_th[i - 1])

            bboxes, score = bboxes_in[mask], score[mask]
            if score.size(0) == 0:
                continue

            _, score_idx_sorted = score.sort(dim=0)

            # select max_num indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = torch.empty(score.size(0), device=score.device).zero_().long()
            count = 0
            while score_idx_sorted.size(0) > 0:
                idx = score_idx_sorted[-1]
                candidates[count] = idx
                count += 1
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze(1)
                # we only need iou < nms_th
                iou_mask = iou_sorted.lt(nms_th)
                score_idx_sorted = score_idx_sorted[iou_mask]
            bboxes_out.append(bboxes[candidates[:count], :])
            scores_out.append(score[candidates[:count]])
            labels_out.append(torch.ones(bboxes[candidates[:count]].shape[0]) * i)

        returns = torch.jit.annotate(List[Tensor], [])

        if not bboxes_out:
            returns.append(torch.empty((0, 4)))
            returns.append(torch.empty((0)))
            returns.append(torch.empty((0)))
        else:
            bboxes_out_t, labels_out_t, scores_out_t = (
                torch.cat(bboxes_out, dim=0),
                torch.hstack(labels_out),
                torch.cat(scores_out, dim=0),
            )

            returns.append(bboxes_out_t)
            returns.append(labels_out_t)
            returns.append(scores_out_t)

        return returns


class DefaultBoxes(object):
    """Default Boxes generation for decoding model outputs."""

    def __init__(
        self,
        fig_size: int,
        feat_size: List[int],
        steps: List[int],
        scales: List[int],
        aspect_ratios: List[List[int]],
        scale_xy: float = 0.1,
        scale_wh: float = 0.2,
    ):
        """Initialise default boxes.

        Parameters
        ----------
        fig_size : int
            Square dimension of the input image the models expect.
        feat_size : List[int]
            List of feature sizes
        steps : List[int]
            List of steps
        scales : List[int]
            List of scales
        aspect_ratios : List[List[int]]
            List of aspect rations
        scale_xy : float, optional
            scale for xy, by default 0.1
        scale_wh : float, optional
            scale for wh, by default 0.2
        """
        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self) -> float:
        """Return the `scale_xy`.

        Returns
        -------
        scale_xy: float
            Scale for xy
        """
        return self.scale_xy_

    @property
    def scale_wh(self) -> float:
        """Return the `scale_wh`.

        Returns
        -------
        scale_wh: float
            Scale for wh
        """
        return self.scale_wh_

    def __call__(self, order: str = "ltrb") -> torch.Tensor:
        """Call to return dboxes in given order

        Parameters
        ----------
        order : str, optional
            Order of the dboxes. One of `"ltrb"` or `"xywh"`, by default "ltrb"

        Returns
        -------
        dboxes : torch.Tensor
            dboxes in the given order
        """
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes
