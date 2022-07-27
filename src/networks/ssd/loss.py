from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import BoxCoder


class SSDDefaultLoss(nn.Module):
    """SSDDefaultLoss is the base loss class for SSD.
    It computs the bounding box classification and regression loss
    for all the images in the batch.

    Parameters
    ----------
    neg_to_pos_ratio: int
        The ratio of negative boxes to positive boxes.
    """

    def __init__(self, neg_to_pos_ratio: int = 3) -> None:
        super().__init__()
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def _setup_box_encoder(self, encoder: BoxCoder):
        self.encoder = encoder

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: List[Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        bbox_regression = head_outputs[0]
        cls_logits = head_outputs[1]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                foreground_matched_idxs_per_image
            ]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.encoder.encode_single(
                matched_gt_boxes_per_image, anchors_per_image
            )
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(
                    bbox_regression_per_image, target_regression, reduction="sum"
                )
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none"
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float(
            "inf"
        )  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "regression_loss": bbox_loss.sum() / N,
            "classification_loss": (
                cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()
            )
            / N,
        }


class CFSSDDefaultLoss(nn.Module):
    """CFSSDDefaultLoss is the base loss class for CFSSD.
    The loss computes the object detection loss and the validation head loss

    Parameters
    ----------
    neg_to_pos_ratio: int, default: 3
        The ratio of negative boxes to positive boxes.
    min_neg_samples: int, default: 0
        The minimum number of negative samples.
    validation_reduction: str, default: "mean"
        The reduction method for the validation loss.
    detection_loss_weight: float, default: 0.5
        The weight of the detection loss.
    validation_loss_weight: float, default: 0.5
        The weight of the validation loss.
    """

    def __init__(
        self,
        neg_to_pos_ratio: int = 3,
        min_neg_samples: int = 0,
        validation_reduction: str = "mean",
        detection_loss_weight: float = 0.5,
        validation_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.min_neg_samples = min_neg_samples
        self.validation_reduction = validation_reduction
        self.detection_loss_weight = detection_loss_weight
        self.validation_loss_weight = validation_loss_weight

    def _setup_box_encoder(self, encoder: BoxCoder):
        self.encoder = encoder

    def _compute_validation_head_loss(
        self, rejection_logits: Tensor, target: List[Tensor], reduction: str = "mean"
    ) -> Tensor:
        # Convert the image_label targets from List[Dict[str, Tensor]] to (N, ) Shaped Tensor

        image_labels = [target_per_image["image_label"].unsqueeze(0) for target_per_image in target]
        image_labels = torch.cat(image_labels, dim=0)
        # If rejection_logits is a (2, ), then unsqueeze it to (1, 2)
        if rejection_logits.dim() == 1:
            rejection_logits = rejection_logits.unsqueeze(0)
        return F.cross_entropy(rejection_logits, image_labels.long(), reduction=reduction)

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        rejection_logits: Tensor,
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        bbox_regression = head_outputs[0]
        cls_logits = head_outputs[1]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                foreground_matched_idxs_per_image
            ]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.encoder.encode_single(
                matched_gt_boxes_per_image, anchors_per_image
            )
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(
                    bbox_regression_per_image, target_regression, reduction="sum"
                )
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none"
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # Replace the 0s with the minimum number of negative samples
        num_negative = torch.where(num_negative == 0, self.min_neg_samples, num_negative)

        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float(
            "inf"
        )  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        # extra samples due to self.min_neg_samples
        N = max(1, num_foreground)
        # images without any boxes
        images_without_boxes = (cls_targets.sum(1) == 0).sum().item()
        N_neg = self.min_neg_samples * images_without_boxes
        regression_loss = bbox_loss.sum() / N
        # select background loss from cls_loss based on background_idxs
        classification_loss = (
            cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()
        ) / (N + N_neg)
        validation_loss = self._compute_validation_head_loss(
            rejection_logits, targets, reduction=self.validation_reduction
        )
        loss = (
            self.detection_loss_weight * (regression_loss + classification_loss)
            + self.validation_loss_weight * validation_loss
        )

        return {
            "loss": loss,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            "validation_loss": validation_loss,
        }


class CFSSDLossWithLabelSmoothing(CFSSDDefaultLoss):
    """
    This is a modified version of the default CFSSD loss function.
    We use label smoothing on the validation head
    """

    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _compute_validation_head_loss(
        self, rejection_logits: Tensor, target: List[Tensor], reduction: str = "mean"
    ) -> Tensor:
        # Convert the image_label targets from List[Dict[str, Tensor]] to (N, ) Shaped Tensor

        image_labels = [target_per_image["image_label"].unsqueeze(0) for target_per_image in target]
        image_labels = torch.cat(image_labels, dim=0).long()
        # If rejection_logits is a (2, ), then unsqueeze it to (1, 2)
        if rejection_logits.dim() == 1:
            rejection_logits = rejection_logits.unsqueeze(0)

        # Label smoothing
        n = rejection_logits.size()[-1]
        log_preds = F.log_softmax(rejection_logits, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), reduction)
        nll = F.nll_loss(log_preds, image_labels, reduction=reduction)
        return self.epsilon * (loss / n) + (1 - self.epsilon) * nll


def reduce_loss(loss, reduction="mean"):
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
