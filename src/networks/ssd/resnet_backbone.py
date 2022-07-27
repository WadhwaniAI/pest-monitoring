""" Modifying the ResNet Backbone to create a SSDFeatureExtractorResNet() """

from collections import OrderedDict
from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet

from .utils import _validate_trainable_layers, _xavier_init

model_urls = {
    "ssd512_resnet50_coco": "https://download.pytorch.org/models/ssd512_resnet50_coco-d6d7edbb.pth",
}


def _resnet_extractor(
    backbone_name: str = "resnet18",
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
) -> nn.Module:
    """Returns a resnet extractor with the given backbone name, pretrained, and trainable layers.

    Parameters
    ----------
    backbone_name: str
        Resnet backbone name, can be resnet18/34/50/101/152
    pretrained: bool
        If True, returns a model pre-trained on COCO train2017
    pretrained_backbone: bool
        If True, returns a model with backbone pre-trained on Imagenet
    trainable_backbone_layers: Optional[int]
        Number of backbone layers to train.
    """
    if pretrained:
        raise NotImplementedError("Pretrained not supported for SSD")
        pretrained_backbone = False

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5
    )

    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone)

    assert 0 <= trainable_backbone_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_backbone_layers]
    if trainable_backbone_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return SSDFeatureExtractorResNet(backbone)


class SSDFeatureExtractorResNet(nn.Module):
    """SSD Feature Extractor that uses a ResNet backbone."""

    def __init__(self, backbone: resnet.ResNet) -> None:
        """Initialize the SSD Feature Extractor.

        Parameters
        ----------
        backbone: ResNet
            ResNet backbone to use
        """
        super().__init__()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # Patch last block's strides to get valid output sizes
        for m in self.features[-1][0].modules():
            if hasattr(m, "stride"):
                m.stride = 1

        # TODO: find a better way for only resnet50 and above
        if hasattr(backbone.layer4[-1], "bn3"):
            backbone_out_channels = self.features[-1][-1].bn3.num_features
        else:
            backbone_out_channels = self.features[-1][-1].bn2.num_features

        extra = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(backbone_out_channels, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=2, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        _xavier_init(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass of the SSD Feature Extractor."""
        x = self.features(x)
        output = [x]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


final_output_size = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


class CFSSDResNetRejectionHead(nn.Module):
    """CFSSD Rejection Head that should use the backbone"""

    def __init__(
        self, backbone_name: str, validation_classes: int, pretrained: bool = False
    ) -> None:
        """Initialize the CFSSD Rejection Head.
        TODO: Fix this to extend from like CFSSDVGGRejectionHead

        Parameters
        ----------
        backbone_name: str
            Resnet backbone name, can be resnet18/34/50/101/152
        validation_classes: int
            Number of classes in the dataset
        pretrained: bool
            If True, returns a model pre-trained on ImageNet
        """
        super().__init__()
        assert backbone_name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ], "Backbone name must be resnet18/34/50/101/152"
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained)

        # the feature extractor would contain all layers after backbone.layer3
        self.feature_extractor = nn.Sequential(*list(backbone.children())[7:-1])
        self.fc_layer = nn.Linear(final_output_size[backbone_name], validation_classes)

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward pass of the CFSSD Rejection Head."""
        # currently we only use the first feature map from the feature extractor
        x = self.feature_extractor(x[0]).squeeze()
        return self.fc_layer(x)
