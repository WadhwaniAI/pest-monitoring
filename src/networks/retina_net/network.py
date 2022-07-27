"""RetinaNet implementation inspired by https://github.com/yhenon/pytorch-retinanet."""
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

from . import (
    Anchors,
    BasicBlock,
    BBoxTransform,
    Bottleneck,
    ClassificationModel,
    ClipBoxes,
    PyramidFeatures,
    RegressionModel,
)
from .utils import layer_dict, model_urls


class RetinaNet(nn.Module):
    """RetinaNet: Object Detection Network"""

    def __init__(
        self,
        resnet_backbone: str,
        num_classes: int,
        loss: nn.Module,
        pretrained: bool = False,
        nms_threshold: float = 0.5,
        conf_threshold: float = 0.05,
    ):
        """Init function

        Parameters
        ----------
        resnet_backbone : str
            One of ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        num_classes : int
            Number of object classes
        loss : nn.Module
            Loss function
        pretrained : bool
            Whether to load pretrained weights for resnet backbone
        """
        super(RetinaNet, self).__init__()
        # setup resnet
        self._setup_resnet(resnet_backbone, pretrained)
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold

        layers = layer_dict[resnet_backbone]
        block = BasicBlock if resnet_backbone in ["resnet18", "resnet34"] else Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv2.out_channels,
                self.layer3[layers[2] - 1].conv2.out_channels,
                self.layer4[layers[3] - 1].conv2.out_channels,
            ]
        elif block == Bottleneck:
            fpn_sizes = [
                self.layer2[layers[1] - 1].conv3.out_channels,
                self.layer3[layers[2] - 1].conv3.out_channels,
                self.layer4[layers[3] - 1].conv3.out_channels,
            ]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.loss = loss

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _setup_resnet(self, backbone_name: str, pretrained: bool):
        if backbone_name not in layer_dict:
            raise ValueError(f"Backbone {backbone_name} not understood")
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls[backbone_name]), strict=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

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

        return classification, regression, anchors

    def forward(self, images, targets):
        """Forward pass of the model"""
        classification, regression, anchors = self.pre_forward(images)

        if targets is None:
            return classification, regression, anchors

        return self.loss(classification, regression, anchors, targets)

    def predict(self, images: torch.Tensor):
        """This step is used to predict given an image"""
        detections = []
        # compute pre-forward pass for predictions
        with torch.no_grad():
            classification_, regression_, anchors = self.pre_forward(images)

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
                }
            )

        return detections
