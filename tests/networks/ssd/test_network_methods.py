"""Tests methods of src.networks.ssd.SSD"""
import unittest
from os.path import exists, join

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf as oc


class SSDMethodsTests(unittest.TestCase):
    """Class to check the Instantiations of SSD"""

    @classmethod
    def setUpClass(cls):
        cls.base_path = "tests/helpers/resources/configs/model/network/ssd/"
        cls.targets = [{"boxes": torch.randn(2, 4), "labels": torch.randint(1, 3, (2,)).long()}]

        if not exists(join(cls.base_path, "base.yaml")):
            raise FileNotFoundError("base.yaml file not found in {}".format(cls.base_path))

    def test_forward_without_target(self):
        """Test SSD Forward without target passed"""
        try:
            network = self._get_network()
            _ = network(torch.randn(1, 3, 512, 512))
        except AssertionError:
            print("Forward without target failed")

    def test_forward_output_without_target(self):
        """Test SSD Forward without target passed"""
        network = self._get_network()
        output = network(torch.randn(1, 3, 512, 512))

        self.assertTrue(output is not None)
        self.assertTrue(len(output) == 3, "Output should have 3 values")

        # check if shape of plocs, plabels match
        plocs, plabels, anchors = output
        self._test_plocs_plabels_shape(plocs, plabels, anchors[0], 3)

    def test_forward_with_target(self):
        """Test SSD Forward with targets passed"""
        try:
            network = self._get_network()
            _ = network(torch.randn(1, 3, 512, 512), self.targets)
        except AssertionError:
            print("Forward with target failed")

    def test_forward_output_with_target(self):
        """Test SSD Forward with targets passed"""
        network = self._get_network()
        output = network(torch.randn(1, 3, 512, 512), self.targets)

        self.assertTrue(output is not None)
        self.assertTrue(type(output) == dict, "Output should be a dict")
        # assert that ['regression_loss', 'classification_loss'] in output
        self.assertTrue("regression_loss" in output, "regression_loss should be in output")
        self.assertTrue("classification_loss" in output, "classification_loss should be in output")

    def test_predict(self):
        """Test predict method"""
        network = self._get_network()
        try:
            _ = network.predict(torch.randn(1, 3, 512, 512), [[512, 512]])
        except AssertionError:
            print("Predict failed")

    def test_predict_output(self):
        """Test predict method output"""
        network = self._get_network()
        output = network.predict(torch.randn(1, 3, 512, 512), [[512, 512]])
        self.assertTrue(output is not None)
        self.assertTrue(type(output) == list, "Output should be a list")
        self.assertTrue(type(output[0]) == dict, "Output[0] should be a dict")
        # assert that boxes, labels and scores are in the dict keys of the first element
        self.assertTrue(
            "boxes" in output[0], "boxes should be in the dict keys of the first element"
        )
        self.assertTrue(
            "labels" in output[0], "labels should be in the dict keys of the first element"
        )
        self.assertTrue(
            "scores" in output[0], "scores should be in the dict keys of the first element"
        )

    def _test_plocs_plabels_shape(
        self, plocs: torch.Tensor, plabels: torch.Tensor, anchors: torch.Tensor, num_classes: int
    ):
        """Tests shape of plocs and plabels"""
        self.assertTrue(
            plocs.detach().numpy().shape == (1, 24692, 4),
            "Expected shape of plocs : (1, 24692, 4), Actual shape of plocs :"
            f" {plocs.detach().numpy().shape}",
        )
        self.assertTrue(
            plabels.detach().numpy().shape == (1, 24692, num_classes),
            "Expected shape of plabels : (1, 24692, num_classes), Actual shape"
            f" of plabels : {plabels.detach().numpy().shape}",
        )
        self.assertTrue(
            anchors.detach().numpy().shape == (24692, 4),
            "Expected shape of anchors : (24692, 4), Actual shape of anchors :"
            f" {anchors.detach().numpy().shape}",
        )

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
