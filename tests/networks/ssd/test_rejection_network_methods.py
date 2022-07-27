"""Tests methods of src.networks.ssd.SSD"""
import unittest
from os.path import exists, join

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf as oc


class SSDMethodsTests(unittest.TestCase):
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
        cls.targets = [
            {
                "boxes": torch.randn(2, 4),
                "labels": torch.randint(1, 3, (2,)).long(),
                "image_label": torch.randint(0, 2, (1, 1)).squeeze().long(),
            }
        ]

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

        # check if shape of plocs, plabels match
        plocs, plabels, rejection_logits, anchors = output
        self._test_output_shape(plocs, plabels, rejection_logits, anchors[0], 3)

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
        # assert that ['regression_loss', 'classification_loss', 'validation_loss'] in output
        self.assertTrue("loss" in output, "loss should be in output")
        self.assertTrue("regression_loss" in output, "regression_loss should be in output")
        self.assertTrue("classification_loss" in output, "classification_loss should be in output")
        self.assertTrue("validation_loss" in output, "validation_loss should be in output")

    def test_predict(self):
        """Test predict method"""
        network = self._get_network()
        try:
            _ = network.predict(torch.randn(1, 3, 512, 512))
        except AssertionError:
            print("Predict failed")

    def test_predict_output(self):
        """Test predict method output"""
        network = self._get_network()
        output = network.predict(torch.randn(1, 3, 512, 512))
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
        self.assertTrue(
            "validation_scores" in output[0],
            "validation_scores should be in the dict keys of the first element",
        )

    def _test_output_shape(
        self,
        plocs: torch.Tensor,
        plabels: torch.Tensor,
        rejection_logits: torch.Tensor,
        anchors: torch.Tensor,
        num_classes: int,
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
            rejection_logits.detach().numpy().shape == (2,),
            "Expected shape of rejection_logits : (2, ), Actual shape of"
            f" rejection_logits : {rejection_logits.detach().numpy().shape}",
        )
        self.assertTrue(
            anchors.detach().numpy().shape == (24692, 4),
            "Expected shape of anchors : (24692, 4), Actual shape of anchors :"
            f" {anchors.detach().numpy().shape}",
        )

    def _get_network(self, backbone: str = "resnet"):
        """Returns network"""
        network_config = oc.load(join(self.base_path, f"base-with-reject-{backbone}.yaml"))
        network = instantiate(network_config)
        return network


if __name__ == "__main__":
    unittest.main()
