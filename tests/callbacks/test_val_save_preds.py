"""Tests src.callbacks.BinaryValSavePreds"""
import json
import unittest
import warnings
from os.path import exists
from shutil import rmtree

import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

warnings.filterwarnings("ignore")


class BinaryValCallbackTestCase(unittest.TestCase):
    """TestCase for src.callbacks.BinaryValSavePreds"""

    @classmethod
    def setUpClass(cls):
        """Set up the class"""
        # Initialize the LightningModule
        config_name = "tests/helpers/resources/configs/model/classification/test-binary.yaml"
        learning_rate = 0.001  # random learning rate
        cls.model_args = OmegaConf.load(config_name)
        cls.model: LightningModule = instantiate(
            cls.model_args.model,
            learning_rate=learning_rate,
            model_config=cls.model_args.model,
            _recursive_=False,
        )

        # Load BaseDatamodule and json using config
        config_name = "tests/helpers/resources/configs/datamodule/test-bin-clf-datamodule.yaml"
        cls.datamodule_args = OmegaConf.load(config_name)

        # Initialize the BaseDatamodule
        cls.datamodule: LightningDataModule = instantiate(
            cls.datamodule_args,
            data_config=cls.datamodule_args,
            shuffle=False,
            _recursive_=False,
        )
        cls.datamodule.setup()
        cls.train_dataloader = cls.datamodule.train_dataloader()
        cls.val_dataloader = cls.datamodule.val_dataloader()
        cls.test_dataloader = cls.datamodule.test_dataloader()
        cls.batch_size = cls.datamodule_args.batch_size

    def test_train_split_callback(self):
        """Test the callback on train split"""
        self._run_callback_any_split("train")
        json_path = "test-callback/pred_train_test-image-bin-clf-single-head-file_0.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the output json file is correct
        self._check_json_file(json_file, "train")

    def test_val_split_callback(self):
        """Test the callback on val split"""
        self._run_callback_any_split("val")
        json_path = "test-callback/pred_val_test-image-bin-clf-single-head-file_0.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the output json file is correct
        self._check_json_file(json_file, "val")

    def test_test_split_callback(self):
        """Test the callback on test split"""
        self._run_callback_any_split("test")
        json_path = "test-callback/pred_test_test-image-bin-clf-single-head-file_0.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)

        # check if the output json file is correct
        self._check_json_file(json_file, "test")

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

        Raises
        ------
        ValueError
            If the split is not "train", "val" or "test"
        """
        # setup callback
        callback_config_name = (
            "tests/helpers/resources/configs/callbacks"
            f"/save-preds/binary-val-save-pred-{split}.yaml"
        )
        callback_config = OmegaConf.load(callback_config_name)
        callbacks = [
            instantiate(callback_config, run_dict=self._get_run_dict(split), _recursive_=False)
        ]

        # setup trainer
        trainer = Trainer(
            default_root_dir=".",
            max_epochs=1,
            gpus=None,
            callbacks=callbacks,
            logger=False,
            checkpoint_callback=False,
        )

        # running the model as "val" mode with different dataloaders
        if split == "train":
            trainer.validate(self.model, self.train_dataloader)
        elif split == "val":
            trainer.validate(self.model, self.val_dataloader)
        elif split == "test":
            trainer.validate(self.model, self.test_dataloader)
        else:
            raise ValueError(f"{split} is not a valid split")

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
        if info["version"] != "test-image-bin-clf-single-head-file":
            raise ValueError("version is not correct")

        if info["split"] != split:
            raise ValueError("split is not correct")

        if info["contributor"] != "test-user":
            raise ValueError("user is not correct")

        if info["ckpt_path"] != "/tmp":
            raise ValueError("ckpt_path is not correct")

        if info["url"] != f"test-callback/pred_{split}_test-image-bin-clf-single-head-file_0.json":
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

        # check if box annotations are empty
        if len(json_file["box_annotations"]) != 0:
            raise ValueError("box annotations should be empty")

        # check if images and splits have equal number of images
        if len(json_file["images"]) != len(json_file["splits"]):
            raise ValueError("images and spits have different number of images")

        # check if input and output jsons have the same images
        if split == "train":
            dl = self.train_dataloader
        elif split == "val":
            dl = self.val_dataloader
        elif split == "test":
            dl = self.test_dataloader
        else:
            raise ValueError(f"{split} is not a valid split")
        gt_img_ids = []
        for batch in dl:
            gt_img_ids.extend(batch["img_id"])
        gt_img_ids = set(gt_img_ids)
        pred_df = pd.DataFrame(json_file["images"])
        pred_img_ids = set(pred_df["id"].values.tolist())
        self.assertEqual(pred_img_ids, gt_img_ids)

        @classmethod
        def tearDownClass(cls) -> None:
            # remove the folders and it's contents created during the tests
            path = "test-callback/"
            if exists(path):
                rmtree(path)


if __name__ == "__main__":
    unittest.main()
