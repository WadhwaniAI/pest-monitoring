"""Tests Optimizers used in Pest-Monitoring-New"""
import unittest

import hydra


class OptimizerTestCase(unittest.TestCase):
    """Class to check the creation of Optimizer"""

    @classmethod
    def setUpClass(cls):
        network_args = {"_target_": "src.networks.ssd_nvidia.SSD300", "num_classes": 3}
        cls.network = hydra.utils.instantiate(network_args)

    def test_adam(self):
        """Test creation of a Adam optmizer"""
        optimizer_name = "torch.optim.Adam"
        optimizer_args = {"_target_": optimizer_name, "lr": 0.0003, "weight_decay": 0.0005}

        _ = hydra.utils.instantiate(optimizer_args, params=self.network.parameters())

    def test_sgd(self):
        """Test creation of a SGD optmizer"""
        optimizer_name = "torch.optim.SGD"
        optimizer_args = {"_target_": optimizer_name, "lr": 0.0003, "momentum": 0.9}

        _ = hydra.utils.instantiate(optimizer_args, params=self.network.parameters())


if __name__ == "__main__":
    unittest.main()
