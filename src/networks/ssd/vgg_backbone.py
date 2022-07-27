""" Modifying the VGG Backbone to create a SSDFeatureExtractorVGG() """

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torchvision.models import vgg

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .utils import _validate_trainable_layers, _xavier_init

backbone_urls = {
    "vgg16_features": "https://download.pytorch.org/models/vgg16_features-amdegroot.pth"
}


def _vgg_extractor(
    backbone_name: str = "vgg16",
    highres: bool = False,
    progress: bool = True,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
) -> nn.Module:
    """Returns a VGG Backbone for SSDFeatureExtractor

    Parameters
    ----------
    backbone_name: str
        Name of the VGG backbone to be used, options are vgg16/19
    highres: bool
        Adds an additional Module in the extra to allow for high resolution
    progress: bool
        Whether to display progress bars for downloads
    pretrained: bool
        If True, returns a model pre-trained on COCO train2017
    pretrained_backbone: bool
        If True, returns a model with backbone pre-trained on Imagenet
    trainable_backbone_layers: int
        number of trainable (not frozen) resnet layers starting from final block.
        Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 5
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        raise NotImplementedError("Pretrained COCO backbone for VGG not supported")
        pretrained_backbone = False

    if backbone_name in backbone_urls:
        # Use custom backbones more appropriate for SSD
        arch = backbone_name.split("_")[0]
        backbone = vgg.__dict__[arch](pretrained=False, progress=progress).features
        if pretrained_backbone:
            state_dict = load_state_dict_from_url(backbone_urls[backbone_name], progress=progress)
            backbone.load_state_dict(state_dict)
    else:
        # Use standard backbones from TorchVision
        backbone = vgg.__dict__[backbone_name](
            pretrained=pretrained_backbone, progress=progress
        ).features

    # Gather the indices of maxpools. These are the locations of output blocks.
    stage_indices = [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages

    return SSDFeatureExtractorVGG(backbone, highres)


class L2Norm(nn.Module):
    """L2Norm Layer used in SSD Original Paper"""

    def __init__(self, n_channels, scale) -> None:
        """L2Norm Initializer

        Parameters
        ----------
        n_channels: int
            Number of channels in the input tensor
        scale: float
            Scaling factor for the output tensor
        """
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights"""
        init.constant_(self.weight, self.gamma)

    def forward(self, x) -> Tensor:
        """Forward pass of the layer"""
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSDFeatureExtractorVGG(nn.Module):
    """SSD Feature Extractor based on VGG Backbone"""

    def __init__(self, backbone: nn.Module, highres: bool) -> None:
        """Initializes the SSD Feature Extractor for VGG Backbone

        Parameters
        ----------
        backbone_name: str
            Name of the VGG backbone to be used, options are vgg16/19
        highres: bool
            Adds an additional Module in the extra to allow for high resolution
        """
        super().__init__()

        _, _, maxpool3_pos, maxpool4_pos, _ = (
            i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d)
        )

        # Patch ceil_mode for maxpool3 to get the same WxH output sizes as the paper
        backbone[maxpool3_pos].ceil_mode = True

        # parameters used for L2 regularization + rescaling
        self.L2Norm = L2Norm(512, 20)

        # Multiple Feature maps - page 4, Fig 2 of SSD paper
        self.features = nn.Sequential(*backbone[:maxpool4_pos])  # until conv4_3

        # SSD300 case - page 4, Fig 2 of SSD paper
        extra = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # conv8_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # conv9_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        if highres:
            # Additional layers for the SSD512 case. See page 11, footernote 5.
            extra.append(
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                    nn.ReLU(inplace=True),
                )
            )
        _xavier_init(extra)

        fc = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=1, padding=1, ceil_mode=False
            ),  # add modified maxpool5
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6
            ),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),  # FC7
            nn.ReLU(inplace=True),
        )
        _xavier_init(fc)
        extra.insert(
            0,
            nn.Sequential(
                *backbone[maxpool4_pos:-1],
                fc,
            ),  # until conv5_3, skip maxpool5
        )
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass of the SSD Feature Extractor

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, C, H, W)
        """
        # L2 regularization + Rescaling of 1st block's feature map
        x = self.features(x)
        rescaled = self.L2Norm(x)
        output = [rescaled]

        # Calculating Feature maps for the rest blocks
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


class CFSSDVGGRejectionHead(nn.Module):
    """CFSSD Rejection Head for Image Validation"""

    def __init__(self, num_classes: int = 2):
        """Initializes the CFSSD Rejection Head for VGG Backbone

        Parameters
        ----------
        num_classes: int
            Number of image validation classes to be predicted
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            *[
                # image validation
                nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.AdaptiveAvgPool2d(output_size=(5, 5)),
            ]
        )
        self.fc_layer = nn.Sequential(
            *[
                nn.Linear(in_features=6400, out_features=1024, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=1024, out_features=num_classes, bias=True),
            ]
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward pass of the CFSSD Rejection Head

        Parameters
        ----------
        x: List[Tensor]
            List of feature maps from the feature extractor
        """
        x = self.feature_extractor(x[1])
        x = torch.flatten(x, 1)
        return self.fc_layer(x)
