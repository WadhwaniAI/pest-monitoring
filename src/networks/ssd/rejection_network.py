"""Inspired by the code from SSD Code form
https://github.com/pytorch/vision/

The CFSSD (Rejection Based SSD) class allows for all resnet18/34/50/101/152 backbones
with the image size options of (300, 300) and (512, 512). It does
Object Detection + Classification

"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf as oc
from torch import Tensor
from torchvision.ops import boxes as box_ops

from . import SSDHead
from .utils import BoxCoder, DefaultBoxGenerator, SSDMatcher, retrieve_out_channels


class CFSSD(nn.Module):
    """CFSSD Network: ClassiFication + SSD
    Built upon the SSD Network with the addition of a classification head
    used for image validation.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        anchor_generator: DefaultBoxGenerator,
        rejection_head: nn.Module,
        loss: nn.Module,
        img_size: Tuple[int, int],
        num_classes: int,
        head: Optional[nn.Module] = None,
        iou_thresh: float = 0.5,
        nms_threshold: float = 0.45,
        conf_threshold: float = 0.01,
        detections_per_img: int = 200,
        topk_candidates: int = 400,
    ) -> None:
        """The CFSSD Network is constructed using
        1. FeatureExtractor (e.g. SSDFeatureExtractorResNet, SSDFeatureExtractorVGG)
        2. SSDHead (which contains the SSDClassificationHead and SSDRegressionHead)
        3. RejectionHead

        Parameters
        ----------
        feature_extractor : nn.Module
            Backbone after which SSD Head would be applied
        anchor_generator : DefaultBoxGenerator
            Module that has functionality to generate anchor boxes
        rejection_head : nn.Module
            Module that has functionality of the rejection head
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
        self.rejection_head = rejection_head
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
        """Performs a pre-forward pass of the SSD Network
        Returns the following as a dict:
        1. head_outputs
        2. anchors
        3. rejection_logits

        Parameters
        ----------
        images : Tensor
            The input images post transformations
        """
        # forward pass over feature extractor
        features = self.backbone(images)

        # convert the features from OrderedDict to list
        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        # compute rejection head logits
        rejection_logits = self.rejection_head(features)

        return {
            "anchors": anchors,
            "head_outputs": head_outputs,
            "rejection_logits": [rejection_logits],
        }

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Tuple, Dict]:
        """Performs a forward pass of the CFSSD Network

        Parameters
        ----------
        images : Tensor
            The input images post transformations
        targets : Optional[List[Dict[str, Tensor]]], optional
            The targets per image in a list of dicts.

        Returns
        -------
        Either of the following:
        - loss_dict for the CFSSD Network if targets passed
        - returns the output tensors as a tuple if no targets passed
        """

        # pre-forward step
        pre_forward_step_outputs = self.pre_forward(images)

        head_ouputs, anchors, rejection_logits = (
            pre_forward_step_outputs["head_outputs"],
            pre_forward_step_outputs["anchors"],
            pre_forward_step_outputs["rejection_logits"][0],
        )

        if targets is None:
            # running for jit trace, returning tuple of tensors
            return (
                head_ouputs[0],
                head_ouputs[1],
                rejection_logits,
                anchors,
            )

        # compute loss
        loss_dict = self.compute_loss(head_ouputs, anchors, rejection_logits, targets)

        return loss_dict

    def compute_loss(
        self,
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        rejection_logits: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Computes the loss for the SSD Network

        Parameters
        ----------
        head_outputs : Dict[str, Tensor]
            The output tensors from the SSDHead
        anchors : List[Tensor]
            The anchors for the SSDHead
        rejection_logits : Tensor
            The output tensor from the RejectionHead
        targets : List[Dict[str, Tensor]]
            The targets per image in a list of dicts.
        """
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

        return self.loss(targets, head_outputs, anchors, rejection_logits, matched_idxs)

    def predict(self, images: Tensor) -> List[Dict[str, Tensor]]:
        """Performs the NMS and returns the detections.

        Parameters
        ----------
        images : torch.Tensor
            The input images tensor.
        """
        # pre-forward step
        pre_forward_step_outputs = self.pre_forward(images)
        head_outputs, image_anchors, rejection_logits = (
            pre_forward_step_outputs["head_outputs"],
            pre_forward_step_outputs["anchors"],
            pre_forward_step_outputs["rejection_logits"][0],
        )

        bbox_regression = head_outputs[0]
        pred_scores = F.softmax(head_outputs[1], dim=-1)
        rejection_scores = F.softmax(rejection_logits, dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        # convert image_shapes to List[Tuple[int, int]] of size len(pred_scores)
        image_shapes = [(self.img_size[0], self.img_size[1])] * len(pred_scores)

        for i, (boxes, scores, anchors, image_shape) in enumerate(
            zip(bbox_regression, pred_scores, image_anchors, image_shapes)
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
                    "validation_scores": rejection_scores[i],
                }
            )

        return detections
