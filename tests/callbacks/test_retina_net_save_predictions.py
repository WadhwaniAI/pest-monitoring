"""Tests src.callbacks.retina_net.RetinaNetSavePredictions"""
import json
import unittest
import warnings
from os.path import exists
from shutil import rmtree

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

warnings.filterwarnings("ignore")


class CallbackTestCase(unittest.TestCase):
    """TestCase for src.callbacks.retina_net.RetinaNetSavePredictions"""

    @classmethod
    def setUpClass(cls):
        """Set up the class"""
        # Initialize the LightningModule
        config_name = "tests/helpers/resources/configs/model/retina-net/resnet18.yaml"
        cls.model_args = OmegaConf.load(config_name)
        cls.model: LightningModule = instantiate(
            cls.model_args, model_config=cls.model_args, _recursive_=False
        )

        # Load BaseDatamodule using config
        config_name = (
            "tests/helpers/resources/configs/datamodule/"
            "test-object-detection-datamodule-retina-net.yaml"
        )
        cls.datamodule_args = OmegaConf.load(config_name)

        # Initialize the BaseDatamodule
        cls.datamodule: LightningDataModule = instantiate(
            cls.datamodule_args,
            data_config=cls.datamodule_args,
            shuffle=False,
            _recursive_=False,
        )
        cls.datamodule.setup()
        cls.dataloader = {
            "train": cls.datamodule.train_dataloader(),
            "val": cls.datamodule.val_dataloader(),
            "test": cls.datamodule.test_dataloader(),
        }
        cls.batch_size = cls.datamodule_args.batch_size

    def test_train_split_callback(self):
        """Test the callback on train split"""
        self._run_callback_any_split("train")
        json_path = (
            "test-callback/pred_train_test-object-det-file_nms_0.5_conf_threshold_0.05_0.json"
        )
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the json file is correct
        self._check_json_file(json_file, "train")

    def test_val_split_callback(self):
        """Test the callback on val split"""
        self._run_callback_any_split("val")
        json_path = "test-callback/pred_val_test-object-det-file_nms_0.5_conf_threshold_0.05_0.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the json file is correct
        self._check_json_file(json_file, "val")

    def test_test_split_callback(self):
        """Test the callback on test split"""
        self._run_callback_any_split("test")
        json_path = (
            "test-callback/pred_test_test-object-det-file_nms_0.5_conf_threshold_0.05_0.json"
        )
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the json file is correct
        self._check_json_file(json_file, "test")

    def test_convert_xyx2y2_to_xywh(self):
        """Test the conversion from relative x1y1x2y2 to absolute xywh format"""
        x1y1x2y2 = [10, 20, 30, 40]
        xywh = self._convert_xyx2y2_to_xywh(x1y1x2y2)
        self.assertEqual(xywh, [10, 20, 20, 20])

    def _convert_xyx2y2_to_xywh(self, coords):
        """Helper function to convert absolute x1y1x2y2
        to absolute x1y1w2h

        Parameters
        ----------
        coords : list
            List of coordinates in absolute x1y1x2y2 format

        Returns
        -------
        list
            List of coordinates in absolute xywh format
        """
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1
        return [x1, y1, w, h]

    # Helper functions
    def _get_run_dict(self, split: str = "train"):
        """Helper function to create the test run dict

        Parameters
        ----------
        split : str, optional
            Split on which the callback is run, by default "train"

        Returns
        -------
        dict
            The test run dict
        """
        return {
            "ckpt_path": "/tmp",
            "split": split,
            "user": "test-user",
            "config": {
                "datamodule": self.datamodule_args,
                "model": self.model_args,
            },
        }

    def _run_callback_any_split(self, split: str = "train"):
        """Helper function to run callback on any split

        Parameters
        ----------
        split : str, optional
            Split to run callback on, by default "train"
        """
        # setup callback
        callback_config_name = (
            "tests/helpers/resources/configs/callbacks/retina_net_save_preds.yaml"
        )
        callback_config = OmegaConf.load(callback_config_name)
        callbacks = [
            instantiate(callback_config, run_dict=self._get_run_dict(split), _recursive_=False)
        ]

        # setup trainer
        trainer = Trainer(
            default_root_dir=".",
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=1,
            callbacks=callbacks,
            logger=False,
            checkpoint_callback=False,
        )

        # running the model as "val" mode with different dataloaders
        try:
            trainer.validate(self.model, self.dataloader[split])
        except Exception as e:
            self.fail(f"Validation Loop failed with error: {e}")

    def _check_info(self, info: dict, split: str):
        """Check if info is correct

        Parameters
        ----------
        info : dict
            The info dictionary from the json dictionary of the predictions
        split : str
            The split of the predictions

        Raises
        ------
        ValueError
            If version does not match with version in config
        ValueError
            If split does not match the split in function argument
        ValueError
            If user does not match with the run_dict['user']
        ValueError
            If ckpt_path does not match with the run_dict['ckpt_path']
        ValueError
            If url does not match with the run_dict['url']
        """
        if info["version"] != "test-object-det-file":
            raise ValueError("version is not correct")

        if info["split"] != split:
            raise ValueError("split is not correct")

        if info["contributor"] != "test-user":
            raise ValueError("user is not correct")

        if info["ckpt_path"] != "/tmp":
            raise ValueError("ckpt_path is not correct")

        if (
            info["url"]
            != f"test-callback/pred_{split}_test-object-det-file_nms_0.5_conf_threshold_0.05_0.json"
        ):
            raise ValueError("url is not correct")

    def _check_json_file(self, json_file: dict, split: str):
        """Check if json file is correct

        Parameters
        ----------
        json_file : dict
            The json dictionary of the predictions
        split : str
            The split of the predictions

        Raises
        ------
        KeyError
            If the keys ["info", "images", "box_annotations", "caption_annotations", "splits"]
            are not in the json file
        ValueError
            If the caption_annotations have greater than 0 length
        ValueError
            If the length of images and splits do not match
        """
        keys = ["info", "images", "box_annotations", "caption_annotations", "splits"]
        for key in keys:
            if key not in json_file:
                raise KeyError(f"{key} not in json file")

        # check info
        self._check_info(json_file["info"], split)

        # check if caption annotations are empty
        if len(json_file["caption_annotations"]) != 0:
            raise ValueError("caption annotations should be empty")

        # check if images and splits have equal number of images
        if len(json_file["images"]) != len(json_file["splits"]):
            raise ValueError("images and spits have different number of images")

    @classmethod
    def tearDownClass(cls) -> None:
        # remove the folders and it's contents created during the tests
        path = "test-callback/"
        if exists(path):
            rmtree(path)


if __name__ == "__main__":
    unittest.main()
