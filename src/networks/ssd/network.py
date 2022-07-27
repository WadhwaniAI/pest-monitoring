"""Inspired by the code from SSD Code form
https://github.com/pytorch/vision/

The SSD class allows for all resnet18/34/50/101/152 backbones
with the image size options of (300, 300) and (512, 512).

"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf as oc
from torch import Tensor
from torchvision.ops import boxes as box_ops

from .utils import (
    BoxCoder,
    DefaultBoxGenerator,
    SSDMatcher,
    _xavier_init,
    retrieve_out_channels,
)


class SSDScoringHead(nn.Module):
    """Generic SSD scoring head class for regression and classification heads"""

    def __init__(self, module_list: nn.ModuleList, num_columns: int) -> None:
        """Initializes SSDScoringHead

        Parameters
        ----------
        module_list : nn.ModuleList
            The list of modules that make up the scoring head
        num_columns : int
            Number of columns in the output tensor
        """
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.module_list:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Passes input features, x, through layers specified in module list.
        x[i] becomes module_list[i](x[i])
        """
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.reshape(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    """Class to create a Classification Head for the SSD network"""

    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None:
        """Creates a nn.ModuleList to corresponding to classification head in SSD.

        Parameters
        ----------
        in_channels : List[int]
            Input channels in the features that will be passed to Classification layers
        num_anchors : List[int]
            Number of anchor boxes
        num_classes : int
            Number of possible classes of predicted box
        """
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    """Class to create a Regression Head for the SSD network"""

    def __init__(self, in_channels: List[int], num_anchors: List[int]) -> None:
        """Creates a nn.ModuleList to corresponding to regression head in SSD.

        Parameters
        ----------
        in_channels : List[int]
            Input channels in the features that will be passed to Regression layers
        num_anchors : List[int]
            Number of anchor boxes
        """
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSDHead(nn.Module):
    """SSD head for the SSD Model"""

    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None:
        """Initializes regression and classificaiton head of SSD

        Parameters
        ----------
        in_channels : List[int]
            Input channels in the features that will be passed to Classification layers
        num_anchors : List[int]
            Number of anchor boxes
        num_classes : int
            Number of possible classes of predicted box
        """
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Returns a list of output of regression and classification heads on input x"""
        return [self.regression_head(x), self.classification_head(x)]


class SSD(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        anchor_generator: DefaultBoxGenerator,
        loss: nn.Module,
        img_size: int,
        num_classes: int,
        head: Optional[nn.Module] = None,
        iou_thresh: float = 0.5,
        nms_threshold: float = 0.45,
        conf_threshold: float = 0.01,
        detections_per_img: int = 200,
        topk_candidates: int = 400,
    ) -> None:
        """Intializes SSD network. The SSD Network is constructed using
        1. FeatureExtractor (e.g. SSDFeatureExtractorResNet, SSDFeatureExtractorVGG)
        2. SSDHead (which contains the SSDClassificationHead and SSDRegressionHead)

        Parameters
        ----------
        feature_extractor : nn.Module
            Backbone after which SSD Head would be applied
        anchor_generator : DefaultBoxGenerator
            Module that has functionality to generate anchor boxes
        loss : nn.Module
            Loss function
        img_size : int
            Image size
        num_classes : int
            Number of classes
        head : Optional[nn.Module], optional
            Layers applied on top of feature_extractor. If None, `SSDHead` is used, by default None
        iou_thresh : float, optional
            IoU threshold, by default 0.5
        nms_threshold : float, optional
            The NMS threshold, by default 0.45
        conf_threshold : float, optional
            The confidence threshold for the detections, by default 0.01
        detections_per_img : int, optional
            Maximum number of detections per image after NMS, by default 200
        topk_candidates : int, optional
            Maximum number of candidates to be considered before NMS, by default 400
        """
        super().__init__()
        self.backbone = feature_extractor
        self.num_classes = num_classes
        self.anchor_generator = anchor_generator
        self.loss = loss
        self.img_size = oc.to_container(img_size)
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        if head is None:
            if hasattr(feature_extractor, "out_channels"):
                self.out_channels = feature_extractor.out_channels
            else:
                self.out_channels = retrieve_out_channels(feature_extractor, self.img_size)

            assert len(self.out_channels) == len(self.anchor_generator.aspect_ratios)

            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(self.out_channels, num_anchors, self.num_classes)
        self.head = head

        self.encoder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.proposal_matcher = SSDMatcher(iou_thresh)

        # Setup loss with box_coder
        self.loss._setup_box_encoder(self.encoder)

    def pre_forward(self, images: Tensor) -> Dict[str, List[Tensor]]:
        """Performs pre_forward pass using the SSD network
        Returns the following as a dictionary:
        1. head_outputs: The output of the SSDHead
        2. anchors: The anchors generated by the anchor_generator
        """
        # forward pass over feature extractor
        features = self.backbone(images)

        # convert the features from OrderedDict to list
        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        return {
            "anchors": anchors,
            "head_outputs": head_outputs,
        }

    def forward(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]] = None,
    ) -> Union[Tuple, Dict]:
        """Returns
        - the loss for the SSD model if targets passed
        - output tensors as tuples if only images passed
        """
        # perform pre_forward
        pre_forward_outputs = self.pre_forward(images)

        head_outputs, anchors = (
            pre_forward_outputs["head_outputs"],
            pre_forward_outputs["anchors"],
        )

        if targets is None:
            return (head_outputs[0], head_outputs[1], anchors)

        # compute the loss
        loss_dict = self.compute_loss(head_outputs, anchors, targets)

        return loss_dict

    def compute_loss(
        self,
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, float]:
        """Computes the loss for the SSD Network"""
        # Raise an error if targets is none
        if targets is None:
            raise ValueError("No targets provided")

        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.loss(targets, head_outputs, anchors, matched_idxs)

    def predict(self, images: Tensor) -> List[Dict[str, Tensor]]:
        """Performs the NMS and returns the detections.

        Parameters
        ----------
        images : Tensor
            The input images tensor.

        Returns
        -------
        List[Dict[str, Tensor]]
            List with model's output for each image
        """
        # perform pre_forward
        pre_forward_outputs = self.pre_forward(images)
        head_outputs, image_anchors = (
            pre_forward_outputs["head_outputs"],
            pre_forward_outputs["anchors"],
        )

        bbox_regression = head_outputs[0]
        pred_scores = F.softmax(head_outputs[1], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        # convert image_shapes to List[Tuple[int, int]] of size len(pred_scores)
        image_shapes = [(self.img_size[0], self.img_size[1])] * len(pred_scores)

        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, image_anchors, image_shapes
        ):
            boxes = self.encoder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.conf_threshold
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(score, fill_value=label, dtype=torch.int64, device=device)
                )

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_threshold)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections
