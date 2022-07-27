import torch


class ClassificationImageNet(torch.nn.Module):
    """Takes any random CNN backbone and allows any image to be classified into classes.
    The class assumes backbones used for Image Classification on ImageNet.
    That is, # of channels in a image = 3 and last layer having 1000 activations.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone to use (usually picked from `torchvision.models`)
    num_classes : int
        Number of in channels of the images in the dataset
    in_channels : int, optional
        Number of classes for the image to be classified into, by default 3
    """

    def __init__(self, backbone: torch.nn.Module, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.channel_corrector = torch.nn.Conv2d(in_channels, 3, 1)  # image net classification
        self.backbone = backbone
        self.out_layer = torch.nn.Linear(1000, num_classes)  # image net classification

    def forward(self, x):
        x = self.channel_corrector(x)
        x = self.backbone(x)
        x = self.out_layer(x)
        return x
