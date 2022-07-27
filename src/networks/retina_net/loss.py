from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # def __init__(self):
    def forward(self, classifications, regressions, anchors, annotations):
        # store if on cuda
        cuda_mode = classifications.is_cuda
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            # bbox_annotation = annotations[j]
            boxes = annotations[j]["boxes"]
            labels = annotations[j]["labels"]
            bbox_annotation = torch.cat((boxes, labels.unsqueeze(1)), dim=1)

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if cuda_mode:
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(
                anchors[0, :, :], bbox_annotation[:, :4]
            )  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if cuda_mode:
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if cuda_mode:
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(
                targets * torch.log(classification)
                + (1.0 - targets) * torch.log(1.0 - classification)
            )

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if cuda_mode:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()
                )
            else:
                cls_loss = torch.where(
                    torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape)
                )

            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
            )

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if cuda_mode:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                if cuda_mode:
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return {
            "classification_loss": torch.stack(classification_losses).mean(dim=0, keepdim=True),
            "regression_loss": torch.stack(regression_losses).mean(dim=0, keepdim=True),
        }


class CFFocalLoss(FocalLoss):
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

    def forward(self, classifications, regressions, anchors, logits, annotations):
        # compute usual focal loss
        loss_dict = super().forward(classifications, regressions, anchors, annotations)

        # compute validation head loss
        loss_dict.update(
            {
                "validation_head_loss": self._compute_validation_head_loss(
                    logits, annotations, reduction="mean"
                )
            }
        )
        return loss_dict
