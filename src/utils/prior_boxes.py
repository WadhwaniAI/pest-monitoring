import itertools
from math import sqrt

import numpy as np
import torch
from torchvision.ops.boxes import box_convert


class DefaultBoxes(object):
    """Generate default boxes for SSD300, SSD512, SSD-Lite, etc.

    Parameters
    ----------
    fig_size: tuple of int
        Figure size of feature map.
    feat_size: tuple of int
        Feature map size.
    steps: tuple of int
        Step size of feature map.
    scales: tuple of float
        Scales of default boxes.
    aspect_ratios: tuple of float
        Aspect ratios of default boxes.
    scale_xy: float
        Scale of center x and center y, default 0.1.
    scale_wh: float
        Scale of width and height, default 0.2.
    """

    def __init__(
        self,
        fig_size,
        feat_size,
        steps,
        scales,
        aspect_ratios,
        scale_xy=0.1,
        scale_wh=0.2,
    ):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
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
        self.dboxes_ltrb = box_convert(self.dboxes, in_fmt="cxcywh", out_fmt="xyxy")

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        else:  # order == "xywh"
            return self.dboxes


def generate_dboxes(model="ssd"):
    """Generate default boxes for SSD300, SSD512, SSD-Lite, etc.

    Parameters
    ----------
    model : str
        The network model name.
    """
    assert model in [
        "ssd",
        "ssd-512",
        "ssd-lite",
    ], "Model must be either ssd, ssd-lite, ssd-512"
    if model == "ssd":
        figsize = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    elif model == "ssd-512":
        figsize = 512
        feat_size = [64, 32, 16, 8, 4, 2, 1]
        steps = [8, 16, 32, 64, 128, 256, 512]
        scales = [18, 50, 82, 114, 146, 178, 210, 242]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
        dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    elif model == "ssd-lite":
        figsize = 300
        feat_size = [19, 10, 5, 3, 2, 1]
        steps = [16, 32, 64, 100, 150, 300]
        scales = [60, 105, 150, 195, 240, 285, 330]
        aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    else:
        raise NotImplementedError
    return dboxes
