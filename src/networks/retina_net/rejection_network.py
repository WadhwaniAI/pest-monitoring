"""CFRetinaNet implementation inspired by https://github.com/yhenon/pytorch-retinanet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from . import RetinaNet

final_output_size = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


class CFRetinaNet(RetinaNet):
    """CFRetinaNet Implementation combining Object Detection and Image Level Classification"""

    def __init__(
        self,
        validation_classes: int,
        **kwargs,
    ):
        """Init function

        Parameters
        ----------
        validation_classes : int
            Number of object classes
        kwargs : dict
            Keyword arguments passed to RetinaNet
        """
        # initialize AdaptiveAvgPool2d and FC layers as per resnet
        # backbone name from kwargs.get('resnet_backbone')
        super(CFRetinaNet, self).__init__(**kwargs)
        resnet_backbone = kwargs.get("resnet_backbone")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_output_size[resnet_backbone], validation_classes)
        self.validation_classes = validation_classes

    def pre_forward(self, images):
        """This step runs before loss computation and returns classification, regression and
        anchors as output"""
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat(
            [self.classificationModel(feature) for feature in features], dim=1
        )

        anchors = self.anchors(images)

        logits = self.fc(self.avgpool(x4).squeeze())

        # additionally returning logits
        return classification, regression, anchors, logits

    def forward(self, images, targets):
        assert targets is not None, "targets should not be None in training mode"

        classification, regression, anchors, logits = self.pre_forward(images)
        return self.loss(classification, regression, anchors, logits, targets)

    def predict(self, images):
        """Bounding Box Predictions + Image Level Predictions"""
        detections = []
        # compute pre-forward pass for predictions
        with torch.no_grad():
            classification_, regression_, anchors, logits_ = self.pre_forward(images)

        rejection_scores = F.softmax(logits_, dim=-1)

        # for each image
        for i in range(len(images)):
            classification, regression = (
                classification_[i].unsqueeze(0),
                regression_[i].unsqueeze(0),
            )

            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, images[i].unsqueeze(0))

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            # if images are on cuda, then convert to cuda tensors
            cuda_mode = images.is_cuda
            if cuda_mode:
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for j in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, j])

                scores_over_thresh = scores > self.conf_threshold

                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]

                anchors_nms_idx = nms(anchorBoxes, scores, self.nms_threshold)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([j] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([j] * anchors_nms_idx.shape[0])
                if cuda_mode:
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat(
                    (finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue)
                )
                finalAnchorBoxesCoordinates = torch.cat(
                    (finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx])
                )

            detections.append(
                {
                    "boxes": finalAnchorBoxesCoordinates,
                    "labels": finalAnchorBoxesIndexes,
                    "scores": finalScores,
                    "validation_scores": rejection_scores[i],
                }
            )

        return detections
