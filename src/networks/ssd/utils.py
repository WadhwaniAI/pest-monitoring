""" SSD utils, DefaultBoxGenerator
Inspired by
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py
"""
import math
import warnings
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch import Tensor


def _xavier_init(conv: nn.Module) -> None:
    """Xaviers initialization

    Parameters
    ----------
    conv: nn.Module
        The convolutional layer to initialize.
    """
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    """This method retrieves the number of output channels of a specific model.

    Parameters
    ----------
    model: nn.Module
        The model for which we estimate the out_channels.
        It should return a single Tensor or an OrderedDict[Tensor].
    size: Tuple[int, int]
        The size (wxh) of the input.

    Returns
    -------
    out_channels: List[int]
        A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels


class DefaultBoxGenerator(nn.Module):
    """Generates the default boxes of SSD for a set of feature maps and image sizes."""

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ) -> None:
        """Initializes the Generator.

        Parameters
        ----------
        aspect_ratios: (List[List[int]])
            The list of aspect ratios for which default boxes are to be generated.
        min_ratio: float
            The minimum ratio between the height and width of the default box.
        max_ratio: float
            The maximum ratio between the height and width of the default box.
        scales: List[float]
            The list of scales for which default boxes are to be generated.
        steps: List[int]
            The list of steps for which default boxes are to be generated.
        clip: bool
            Whether to clip the coordinates of the default boxes to the image boundaries.
        """
        super().__init__()
        if steps is not None:
            assert len(aspect_ratios) == len(steps)
        self.aspect_ratios = OmegaConf.to_container(aspect_ratios)
        self.steps = OmegaConf.to_container(steps)
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [
                    min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)
                ]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = OmegaConf.to_container(scales)

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(
        self,
        num_outputs: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Generates the default box sizes of the network.

        Parameters
        ----------
        num_outputs: int
            The number of feature maps in the network.
        dtype: torch.dtype
            The data type of the tensor.
        device: torch.device
            The device of the tensor. Default is CPU.
        """
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location."""
        # Estimate num of anchors based on aspect ratios: 2 default boxes
        # + 2 * ratios of feaure map.
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(
        self, grid_sizes: List[List[int]], image_size: List[int], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Generates the default boxes of the network."""
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k, y_f_k = [img_shape / self.steps[k] for img_shape in image_size]
            else:
                y_f_k, x_f_k = f_k

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        s = self.__class__.__name__ + "("
        s += "aspect_ratios={aspect_ratios}"
        s += ", clip={clip}"
        s += ", scales={scales}"
        s += ", steps={steps}"
        s += ")"
        return s.format(**self.__dict__)

    def forward(self, images: Tensor, feature_maps: List[Tensor]) -> List[Tensor]:
        """Generates the default boxes of the network."""
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = images.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        image_sizes = images.shape[0] * [image_size]
        for _ in image_sizes:
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                    dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:],
                ],
                -1,
            )
            dboxes_in_image[:, 0::2] *= image_size[1]
            dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Parameters
    reference_boxes: Tensor
        reference boxes
    proposals: Tensor
        boxes to be encoded
    weights: Tensor[4]
        the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


@torch.jit.script
class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    __name__ = "BoxCoder"

    def __init__(
        self,
        weights: Tuple[float, float, float, float],
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ) -> None:
        """Initializes the BoxCoder.

        Parameters
        ----------
        weights: Tuple[float, float, float, float]
            Weights to be used for the encoding of the bounding boxes.
        bbox_xform_clip: float
            Bounding boxes will be linearly scaled between (0, bbox_xform_clip)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def __name__(self) -> str:
        return "BoxCoder"

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        """Encodes for a batch of proposals and reference boxes.

        Parameters
        ----------
        reference_boxes: List[Tensor]
            A list of reference boxes.
        proposals: List[Tensor]
            A list of proposals.
        """
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode boxes for a single image with respect to some reference boxes.

        Parameters
        ----------
        reference_boxes: Tensor
            reference boxes
        proposals: Tensor
            boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Parameters
        ----------
        rel_codes: Tensor
            encoded boxes
        boxes: Tensor
            reference boxes
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
        ).flatten(1)
        return pred_boxes


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.
    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.
    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(
        self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False
    ) -> None:
        """Initialize matcher.

        Parameters
        ----------
        high_threshold: float
            quality values greater than or equal to this value are candidate matches.
        low_threshold: float
            a lower quality threshold used to stratify
            matches into three levels:
            1) matches >= high_threshold
            2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
            3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
        allow_low_quality_matches: bool
            If True, produce additional matches for predictions that have only
            low-quality match candidates. See set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """Call function

        Parameters
        ----------
        match_quality_matrix: Tensor[float]
            An MxN tensor, containing the pairwise quality between M ground-truth
            elements and N predicted elements.

        Returns
        -------
        matches: Tensor[int64]
            An N tensor where N[i] is a matched gt in [0, M - 1] or a
            negative value indicating that prediction i could not be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images during training"
                )
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images during training"
                )

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(
        self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor
    ) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.

        Parameters
        ----------
        matches: Tensor[int64]
            A vector containing -1, 0, or positive values.
        all_matches: Tensor[int64]
            A vector containing -1, 0, or positive values.
        match_quality_matrix: Tensor[float]
            An MxN tensor, containing the pairwise quality between M ground-truth
            elements and N predicted elements.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class SSDMatcher(Matcher):
    """Matcher for SSD based on anchor similarity."""

    def __init__(self, threshold: float) -> None:
        """Initialize matcher.

        Parameters
        ----------
        threshold: float
            Threshold for positive match.
        """
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """Call function

        Parameters
        ----------
        match_quality_matrix: Tensor[float]
            An MxN tensor, containing the pairwise quality between M ground-truth
            elements and N predicted elements.
        """
        matches = super().__call__(match_quality_matrix)

        # For each gt, find the prediction with which it has the highest quality
        _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(
            highest_quality_pred_foreach_gt.size(0),
            dtype=torch.int64,
            device=highest_quality_pred_foreach_gt.device,
        )

        return matches


def _validate_trainable_layers(
    pretrained, trainable_backbone_layers, max_value, default_value
) -> List[int]:
    """Validate trainable backbone layers.

    Parameters
    ----------
    pretrained: bool
        Whether to use pretrained backbone layers.
    trainable_backbone_layers: list
        List of trainable backbone layers.
    max_value: int
        Maximum value of trainable backbone layers.
    default_value: int
        Default value of trainable backbone layers.
    """
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable"
                .format(max_value)
            )
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers


def resize_boxes(
    boxes: Tensor, original_size: Tuple[int, int], new_size: Tuple[int, int]
) -> Tensor:
    """Resize boxes from original_size to new_size.

    Parameters
    ----------
    boxes: Tensor
        Boxes to resize.
    original_size: Tuple[int, int]
        Size of the image before resizing.
    new_size: Tuple[int, int]
        Size of the image after resizing.
    """
    if boxes.shape == torch.Size([0]):  # when images have no predicted bounding boxes
        return boxes

    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
