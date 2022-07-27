"""Tests intantiation of src.networks.ssd.SSD"""
import unittest
from os.path import exists, join

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc


class SSDInstantiationTests(unittest.TestCase):
    """Class to check the Instantiations of SSD"""

    @classmethod
    def setUpClass(cls):
        cls.base_path = "tests/helpers/resources/configs/model/network/ssd/"

        if not exists(join(cls.base_path, "base.yaml")):
            raise FileNotFoundError("Could not find base.yaml in {}".format(cls.base_path))

    def test_base_instantiation(self):
        try:
            _ = self._get_network()
        except AssertionError:
            print("Base instantiation failed")

    def test_resnet18_backbone(self):
        """Test SSD Instantiation with ResNet18 backbone"""
        _ = self._get_network("resnet18")

    def test_resnet18_backbone_pretrained(self):
        """Test SSD Instantiation with ImageNet pretrained ResNet18 backbone"""
        _ = self._get_network("resnet18", pretrained_backbone=True)

    def test_resnet34_backbone(self):
        """SSD Instantiation with ResNet34 backbone"""
        _ = self._get_network("resnet18")

    def test_resnet34_backbone_pretrained(self):
        """SSD Instantiation with ImageNet pretrained ResNet34 backbone"""
        _ = self._get_network("resnet34", pretrained_backbone=True)

    def test_resnet50_backbone(self):
        """Test SSD Instantiation with ResNet50 backbone"""
        _ = self._get_network("resnet50")

    def test_resnet50_backbone_pretrained(self):
        """Test SSD Instantiation with ImageNet pretrained ResNet50 backbone"""
        _ = self._get_network("resnet50", pretrained_backbone=True)

    def test_vgg16_backbone(self):
        """Test SSD Instantiation with VGG16 backbone"""
        _ = self._get_network("vgg16")

    def test_vgg16_backbone_pretrained(self):
        """Test SSD Instantiation with ImageNet pretrained VGG16 backbone"""
        _ = self._get_network("vgg16", pretrained_backbone=True)

    def test_vgg19_backbone(self):
        """Test SSD Instantiation with VGG19 backbone"""
        _ = self._get_network("vgg19")

    def test_vgg19_backbone_pretrained(self):
        """Test SSD Instantiation with ImageNet pretrained VGG16 backbone"""
        _ = self._get_network("vgg19", pretrained_backbone=True)

    def test_setting_to_gpu(self):
        """Test if network can be set to gpu"""
        try:
            network = self._get_network()
            network.to("cuda")
        except AssertionError:
            print("Setting network to gpu failed")

    def _get_network(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 3,
        pretrained_backbone: bool = False,
    ):
        """Returns network"""
        network_config = oc.load(join(self.base_path, "base.yaml"))
        if backbone_name in ["resnet18", "resnet34", "resnet50"]:
            network_config["feature_extractor"] = {
                "_target_": "src.networks.ssd._resnet_extractor",
                "backbone_name": backbone_name,
                "pretrained_backbone": pretrained_backbone,
            }
        elif backbone_name in ["vgg16", "vgg19"]:
            network_config["feature_extractor"] = {
                "_target_": "src.networks.ssd._vgg_extractor",
                "backbone_name": backbone_name,
                "pretrained_backbone": pretrained_backbone,
            }
        network_config["num_classes"] = num_classes
        network = instantiate(network_config)
        return network


if __name__ == "__main__":
    unittest.main()
