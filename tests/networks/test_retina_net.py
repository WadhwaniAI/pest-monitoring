"""Tests src.networks.retina_net"""
import random
import unittest
import warnings

import torch

from src.networks.retina_net.network import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

warnings.filterwarnings("ignore")


class NetworkTestCase(unittest.TestCase):
    """Class to check the working of RetinaNet Network"""

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 4
        cls.image_size = 1024
        cls.imgs = torch.randn(cls.batch_size, 3, cls.image_size, cls.image_size)
        cls.annotations = [torch.randn(random.randint(1, 10), 5) for _ in range(cls.batch_size)]
        cls.net = resnet18(num_classes=10, pretrained=False)

    def test_setting_to_gpu(self):
        """Test that the network is set to gpu"""
        try:
            self.net.cuda()
        except Exception as e:
            self.fail(f"Network cannot be set to gpu: {e}")

    def test_setting_to_cpu(self):
        """Test that the network is set to cpu"""
        try:
            self.net.cpu()
        except Exception as e:
            self.fail(f"Network cannot be set to cpu: {e}")

    def test_pre_forward(self):
        """Test the pre_forward of the network"""
        try:
            self.net.pre_forward(self.imgs)
        except Exception as e:
            self.fail(f"Network cannot do pre_forward pass: {e}")

    def test_forward(self):
        """Test the forward of the network"""
        try:
            self.net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"Network cannot do forward pass: {e}")

    def test_resnet18(self):
        """Test the resnet18 network"""
        # test instantiation
        try:
            net = resnet18(num_classes=10, pretrained=False)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet18 network cannot do forward pass: {e}")

    def test_resnet34(self):
        """Test the resnet34 network"""
        # test instantiation
        try:
            net = resnet34(num_classes=10, pretrained=False)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet34 network cannot do forward pass: {e}")

    def test_resnet50(self):
        """Test the resnet50 network"""
        # test instantiation
        try:
            net = resnet50(num_classes=10, pretrained=False)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet50 network cannot do forward pass: {e}")

    def test_resnet101(self):
        """Test the resnet101 network"""
        # test instantiation
        try:
            net = resnet101(num_classes=10, pretrained=False)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet101 network cannot do forward pass: {e}")

    def test_resnet152(self):
        """Test the resnet152 network"""
        # test instantiation
        try:
            net = resnet152(num_classes=10, pretrained=False)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet152 network cannot do forward pass: {e}")

    def test_resnet18_pretrained(self):
        """Test the resnet18 network"""
        # test instantiation
        try:
            net = resnet18(num_classes=10, pretrained=True)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet18 network cannot do forward pass: {e}")

    def test_resnet34_pretrained(self):
        """Test the resnet34 network"""
        # test instantiation
        try:
            net = resnet34(num_classes=10, pretrained=True)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet34 network cannot do forward pass: {e}")

    def test_resnet50_pretrained(self):
        """Test the resnet50 network"""
        # test instantiation
        try:
            net = resnet50(num_classes=10, pretrained=True)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet50 network cannot do forward pass: {e}")

    def test_resnet101_pretrained(self):
        """Test the resnet101 network"""
        # test instantiation
        try:
            net = resnet101(num_classes=10, pretrained=True)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet101 network cannot do forward pass: {e}")

    def test_resnet152_pretrained(self):
        """Test the resnet152 network"""
        # test instantiation
        try:
            net = resnet152(num_classes=10, pretrained=True)
        except Exception as e:
            self.fail(f"Network cannot be created: {e}")

        # test forward pass
        try:
            net.forward(self.imgs, self.annotations)
        except Exception as e:
            self.fail(f"resnet152 network cannot do forward pass: {e}")
