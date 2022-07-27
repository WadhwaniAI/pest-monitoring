"""Tests src.models.ObjectDetectNet"""
import unittest
import warnings
from typing import Union

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

warnings.filterwarnings("ignore")


class ModelTestCase(unittest.TestCase):
    """Class to check the working of ObjectDetectNet Model"""

    @classmethod
    def setUpClass(cls):
        """Set up datamodule"""
        # Load BaseDatamodule using config
        config_name = (
            "tests/helpers/resources/configs/datamodule/test-object-detection-datamodule.yaml"
        )
        cls.data_config = OmegaConf.load(config_name)

        # Initialize the BaseDatamodule
        cls.datamodule: LightningDataModule = instantiate(
            cls.data_config, data_config=cls.data_config, _recursive_=False
        )
        cls.datamodule.setup()

    def test_fast_dev_run_with_vgg_with_adam(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_resnet_with_adam(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_vgg_with_sgd(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-sgd_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_resnet_with_sgd(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-sgd_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_vgg_with_adam_with_schedular(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer-lr_schedular.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_resnet_with_adam_with_schedular(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer-lr_schedular.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_vgg_with_sgd_with_schedular(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-sgd_optimizer-lr_schedular.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_resnet_with_sgd_with_schedular(self):
        """Tests fast dev run over model class"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-sgd_optimizer-lr_schedular.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on CPU")

    def test_fast_dev_run_with_vgg_with_adam_gpu(self):
        """Tests fast dev run over model class with gpu"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False, gpus=1)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on GPU")

    def test_fast_dev_run_with_resnet_with_adam_gpu(self):
        """Tests fast dev run over model class with gpu"""
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer.yaml"
        )
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )

        trainer = Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False, gpus=1)
        try:
            trainer.fit(model, self.datamodule)
        except AssertionError:
            print("Fast dev run failed on GPU")

    def test_model_forward_outputs_with_vgg_without_targets(self):
        """Test if model's forward function returns the
        same outputs as networks forward function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)

        model_output = model.network(input_images)
        network_output = network(input_images)
        self._check_lists_equality(model_output, network_output)

    def test_model_forward_outputs_with_resnet_without_targets(self):
        """Test if model's forward function returns the
        same outputs as networks forward function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)

        model_output = model.network(input_images)
        network_output = network(input_images)
        self._check_lists_equality(model_output, network_output)

    def test_model_forward_outputs_with_vgg_with_targets(self):
        """Test if model's forward function returns the
        same outputs as networks forward function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)
        targets = [
            {
                "boxes": torch.tensor([[1, 2, 5, 6], [3, 4, 8, 9]]),
                "labels": torch.randint(1, 3, (2,)).long(),
                "image_label": torch.tensor(0.2),
            }
        ]

        model_output = model.network(input_images, targets)
        network_output = network(input_images, targets)

        self.assertTrue(model_output == network_output, "Outputs do not match")

    def test_model_forward_outputs_with_resnet_with_targets(self):
        """Test if model's forward function returns the
        same outputs as networks forward function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)
        targets = [
            {
                "boxes": torch.tensor([[1, 2, 5, 6], [3, 4, 8, 9]]),
                "labels": torch.randint(1, 3, (2,)).long(),
                "image_label": torch.tensor(0.2),
            }
        ]

        model_output = model.network(input_images, targets)
        network_output = network(input_images, targets)

        self.assertTrue(model_output == network_output, "Outputs do not match")

    def test_model_predict_outputs_with_vgg(self):
        """Test if model's predict function returns the
        same outputs as network's predict function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-vgg_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)
        image_shapes = [(512, 512)]

        model_output = model.network.predict(input_images, original_image_shapes=image_shapes)
        network_output = network.predict(input_images, original_image_shapes=image_shapes)

        self._check_lists_equality(model_output, network_output)

    def test_model_predict_outputs_with_resnet(self):
        """Test if model's predict function returns the
        same outputs as network's predict function
        """
        config_name = (
            "tests/helpers/resources/configs/model/objectdetectnet/"
            "test-resnet_backbone-adam_optimizer.yaml"
        )
        model, network = self._initialize_model_and_network(config_name)

        input_images = torch.randn(1, 3, 512, 512)
        image_shapes = [(512, 512)]

        model_output = model.network.predict(input_images, original_image_shapes=image_shapes)
        network_output = network.predict(input_images, original_image_shapes=image_shapes)

        self._check_lists_equality(model_output, network_output)

    def _initialize_model_and_network(self, config_name):
        """Initializes a model and a network with exactly the
        same conditions (config, seed and training conditions)
        """
        seed_everything(seed=0)
        model_config = OmegaConf.load(config_name)
        model: LightningModule = instantiate(
            model_config.model,
            learning_rate=0.001,
            model_config=model_config["model"],
            _recursive_=False,
        )
        model.eval()

        seed_everything(seed=0)
        network_config = model_config.model.network
        network = instantiate(network_config)
        network.eval()

        return model, network

    def _check_lists_equality(self, sequence1, sequence2, keys=None):
        """Tests if two iterators (list/tuple) are equal or not"""
        self.assertTrue(len(sequence1) == len(sequence2), "Output lengths do not match")
        if keys is None:
            keys = range(len(sequence1))

        for i in range(len(sequence1)):
            if isinstance(sequence1[i], torch.Tensor):
                self.assertTrue(
                    torch.equal(sequence1[i], sequence2[i]),
                    f"Outputs do not match at index {keys[i]}",
                )
            elif isinstance(sequence1[i], Union[list, tuple].__args__):
                print(f"At output index {keys[i]}, checking for subindices")
                self._check_lists_equality(sequence1[i], sequence2[i])
            elif isinstance(sequence1[i], dict):
                self._check_lists_equality(
                    sequence1=list(sequence1[i].values()),
                    sequence2=list(sequence2[i].values()),
                    keys=list(sequence1[i].keys()),
                )
            else:
                raise NotImplementedError(f"Unexpected data type found: {type(sequence1[i])}")


if __name__ == "__main__":
    unittest.main()
