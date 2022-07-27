"""Tests intantiation of src.networks.ssd.SSD"""
import unittest
from os.path import exists, join

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc


class SSDInstantiationTests(unittest.TestCase):
    """Class to check the Instantiations of SSD Rejection Network or CFSSD"""

    @classmethod
    def setUpClass(cls):
        cls.base_path = "tests/helpers/resources/configs/model/network/ssd/"
        if not exists(join(cls.base_path, "base-with-reject-vgg.yaml")):
            raise FileNotFoundError(
                "Config file not found: {}".format(join(cls.base_path, "base-with-reject-vgg.yaml"))
            )
        if not exists(join(cls.base_path, "base-with-reject-resnet.yaml")):
            raise FileNotFoundError(
                "Config file not found: {}".format(
                    join(cls.base_path, "base-with-reject-resnet.yaml")
                )
            )

    def test_base_instantiation(self):
        try:
            _ = self._get_network()
        except AssertionError:
            print("Base instantiation failed")

    def test_resnet_backbone(self):
        """Test SSD Instantiation with ResNet backbone"""
        _ = self._get_network("resnet")

    def test_vgg_backbone(self):
        """Test SSD Instantiation with VGG backbone"""
        _ = self._get_network("vgg")

    def test_setting_to_gpu(self):
        """Test if network can be set to gpu"""
        try:
            network = self._get_network()
            network.to("cuda")
        except AssertionError:
            print("Setting network to gpu failed")

    def _get_network(self, backbone: str = "vgg"):
        """Returns network"""
        network_config = oc.load(join(self.base_path, f"base-with-reject-{backbone}.yaml"))
        network = instantiate(network_config)
        return network


if __name__ == "__main__":
    unittest.main()
