import abc
import datetime
import getpass
import json
import os
from collections import defaultdict
from typing import Any

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils import utils
from utils.json.merge_json import merge_jsons

from .utils import resize_boxes

log = utils.get_logger(__name__)


class AbstractSavePredictions(pl.Callback, metaclass=abc.ABCMeta):
    """AbstractSave class to be used by other concrete callbacks.

    The functions are run in the following order:
    1. on_epoch_start
        a. _init_store_dict # init the dict to store information from each batch
    2. _on_batch_end # append the preds to the dict (to be saved eof epoch)
    3. on_epoch_end
        contains the logic to save the preds to disk.
        a. _save_store_dict # uses self.store_dict and pl_module to save pred json
            to disk
    """

    def __init__(
        self,
        ckpt_path: str = None,
        split: str = "val",
        data_file: str = None,
        save_dir: str = "jsons",
        prefix: str = "prediction",
        merge_gt: bool = False,
        merging_config: str = "/workspace/pest-monitoring-new/utils/json/configs/"
        + "merge_config_gt_pred.yaml",
    ):
        """__init__ method.

        Parameters
        ----------
        ckpt_path : str
            The path to the checkpoint file
        split : str
            The split to be used for the predictions
        data_file : str
            The path to the data file
        save_dir : str
            The directory to save the predictions relative
            to the place where the eval file is run.
        prefix : str
            The prefix to be used for the json file.
        merge_gt : bool
            The flag to merge GT JSON with Prediction JSON.
        merging_config : str
            The path to the JSON merging config. Used only if `merge_gt` is `True`.
        """
        super().__init__()
        # checks inputs
        self._check_input(ckpt_path, split, data_file)

        # checks on save_dir
        self._check_save_dir(save_dir)

        self.ckpt_path = ckpt_path
        self.split = split
        self.data_file = data_file
        self.save_dir = save_dir
        self.prefix = self.set_prefix(prefix)
        self.merge_gt = merge_gt
        self.merging_config = merging_config

        # Save information about the run
        self.callback_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        today = datetime.date.today()
        self.date = today.strftime("%y-%m-%d")

        # Get the json dict, that would be used to create our predictions json
        self.final_json = self._get_json(data_file)

        # Create a dataframe of images to retrieve the image sizes
        # image sizes would be used to convert bounding boxes from relative to absolute
        self.df = pd.DataFrame(self.final_json["images"])

    def set_prefix(self, prefix: str = "prediction") -> str:
        return prefix

    @abc.abstractmethod
    def _init_store_dict(self) -> dict:
        """Helps init store_dict."""
        return NotImplementedError

    @abc.abstractmethod
    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Helper function to be called at the end of each batch.
        This hook appends information to the store_dict."""
        return NotImplementedError

    @abc.abstractmethod
    def _save_store_dict(self, pl_module: pl.LightningModule) -> None:
        """Saves the store dict to json.

        Parameters
        ----------
        pl_module : pl.LightningModule
            pl.LightningModule object for the model
        """
        return

    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Hook function to be called at the start of each predict epoch."""
        self.store_dict = self._init_store_dict()

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Hook function to be called at the end of each batch.
        This hook function append the predictions to the preds dictionary.

        Parameters
        ----------
        trainer : pl.Trainer
            pytorch lightning Trainer object
        pl_module : pl.LightningModule
            pytorch lightning Module object
        outputs : STEP_OUTPUT
            outputs of the model
        batch : Any
            batch of data
        batch_idx : int
            batch index
        dataloader_idx : int
            dataloader index
        """
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        """Hook function to be called at the end of each predict epoch."""
        log.info(f"Saving Predictions for {self.split} Set")
        self._save_store_dict(pl_module)

    def _get_json(self, data_file: str):
        """Get the json file from the config.

        Parameters
        ----------
        data_file : str
            Path to the json data_file file

        Returns
        -------
        dict
            The json file
        """
        with open(data_file) as f:
            json_file = json.load(f)
        return json_file

    def _check_input(self, ckpt_path: str, split: str, data_file: str):
        """Checks if the inputs are valid

        Parameters
        ----------
        ckpt_path : str
            The path to the checkpoint file
        split : str
            The split to be used for the predictions
        data_file : str
            The path to the data file
        """
        if ckpt_path is None or split is None or data_file is None:
            raise ValueError("Please provide the ckpt_path, split and data_file")

    def _check_save_dir(self, save_dir):
        """Checks if the save_dir is valid.
        Creates the save_dir if it does not exist.

        Parameters
        ----------
        save_dir : str
            The directory to save the predictions relative
            to the place where the eval file is run.

        Raises
        ------
        ValueError
            if save_dir is None
        """
        if save_dir is None:
            raise ValueError("Save dir is not valid")
        if not os.path.exists(save_dir):
            log.info(f"Creating directory {save_dir}")
            os.makedirs(save_dir)

    def _convert_to_lists(self, preds):
        """Helper function to convert the predictions to lists.

        Parameters
        ----------
        preds : list[torch.Tensor]
            List of predictions for each image in the batch

        Returns
        -------
        preds: list[list]
            List of predictions detached from the gpu, converted to lists
        """
        return [x.detach().cpu().tolist() for x in preds]


class ObjectDetectionWriter(AbstractSavePredictions):
    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Helper function to be called at the end of each batch. This hook function append the
        predictions to the preds dictionary.

        Parameters
        ----------
        trainer : pl.Trainer
            pytorch lightning Trainer object
        pl_module : pl.LightningModule
            pytorch lightning Module object
        outputs : STEP_OUTPUT
            outputs of the model
        batch : Any
            batch of data
        batch_idx : int
            batch index
        dataloader_idx : int
            dataloader index
        """
        # Get predictions from the model
        detections = self.resize_boxes(
            outputs,
            image_shapes=len(outputs) * [[batch["img"].shape[-2], batch["img"].shape[-1]]],
            original_image_shapes=batch["img_size"],
        )

        boxes = [x["boxes"] for x in detections]
        labels = [x["labels"] for x in detections]
        scores = [x["scores"] for x in detections]

        # extend store_dict with information from current batch
        self.store_dict["boxes"].extend(boxes)
        self.store_dict["labels"].extend(labels)
        self.store_dict["scores"].extend(scores)
        self.store_dict["img_ids"].extend(batch["img_id"])

    def _save_store_dict(self, pl_module: pl.LightningModule):
        """Saves the store dict to json.

        Parameters
        ----------
        pl_module : pl.LightningModule
            pl.LightningModule object for the model

        Raises
        ------
        ValueError
            if the splits across all predictions are not the same
        """
        img_ids = self.store_dict["img_ids"]

        # Check if save dir exists else create it
        os.makedirs(self.save_dir, exist_ok=True)

        # Update final json file
        # Replace box annotations
        self.final_json["box_annotations"] = self.set_box_annotations()
        # Only have images in the imgs and split key of the json file corresponding
        # to the images in the img_ids list
        self.final_json["images"] = [
            img for img in self.final_json["images"] if img["id"] in img_ids
        ]
        self.final_json["splits"] = [
            split for split in self.final_json["splits"] if split["image_id"] in img_ids
        ]
        # Checking if the split of the imgs saved are as requested
        if not all(x["split"] == self.split for x in self.final_json["splits"]):
            raise ValueError("Split of the imgs saved are not as requested")

        # Make caption annotations empty
        self.final_json["caption_annotations"] = self.set_caption_annotations()

        # Save predictions to json file and fill info key of dict with information
        pred_file = self.set_prediction_file_name()

        self.final_json["info"] = {
            "version": self.set_data_file(),
            "description": self.set_description(),
            "contributor": getpass.getuser(),
            "url": pred_file,
            "date_created": self.date,
        }
        log.info(f"Saving predictions to {pred_file}")
        with open(pred_file, "w") as f:
            json.dump(self.final_json, f)

        if self.merge_gt:
            print("Merging GT and Prediction JSONs")
            _ = merge_jsons(
                first_file=self._get_json(self.data_file),
                second_file=self.final_json,
                config_path=self.merging_config,
                dest_filepath=self.set_merged_file_name(),
            )

    def _init_store_dict(self):
        """Returns an empty dict to store predictions from different batches

        Returns
        -------
        dict
            box coordinates, box labels, box confidence scores, box image ids
        """
        return defaultdict(list)

    def _convert_xyx2y2_to_xywh(self, coords):
        """Helper function to convert absolute x1y1x2y2 to absolute x1y1w2h

        Parameters
        ----------
        coords : list
            List of absolute x1y1x2y2 coordinates

        Returns
        -------
        coords : list
            List of absolute x1y1w2h coordinates
        """
        x1, y1, x2, y2 = coords
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]

    def set_data_file(self):
        """Sets data file from the config"""
        return os.path.basename(self.data_file).split(".json")[0]

    def set_prediction_file_name(self):
        """Sets final prediction file_name"""
        return os.path.join(self.save_dir, f"{self.prefix}.json")

    def set_merged_file_name(self):
        """Sets final prediction + GT merged file name"""
        return os.path.join(self.save_dir, f"{self.prefix}+GT.json")

    def set_description(self):
        """Sets description of JSON"""
        return f"Prediction of given model at time {self.callback_time} on {self.date}"

    def set_box_annotations(self):
        """Prepares box annotations for save predictions"""
        if not self._check_box_annotations():
            return []

        boxes = self._convert_to_lists(self.store_dict["boxes"])
        labels = self._convert_to_lists(self.store_dict["labels"])
        scores = self._convert_to_lists(self.store_dict["scores"])
        img_ids = self.store_dict["img_ids"]

        # Modify box annotations and categories of the json file
        # appending the predictions annotations and categories in the json file accordingly
        pred_box_annotations = []

        start_count = 0
        for i in range(len(img_ids)):
            img_id = img_ids[i]
            box = boxes[i]
            label = labels[i]
            score = scores[i]

            for j in range(len(box)):
                # Predicted labels are [1, 2, .. N] where background is 0
                pred_box_annotations.append(
                    {
                        "id": start_count,
                        "image_id": img_id,
                        "bbox": self._convert_xyx2y2_to_xywh(box[j]),
                        "bbox_score": score[j],
                        "category_id": (
                            label[j] - 1
                        ),  # As 0 is background, we need to subtract 1 from the label
                    }
                )
                start_count += 1

        return pred_box_annotations

    def set_caption_annotations(self):
        """Prepares Caption Annotations for System Level Callbacks"""
        if not self._check_caption_annotation():
            return []

        validation_scores = self._convert_to_lists(self.store_dict["validation_scores"])
        img_ids = self.store_dict["img_ids"]

        caption_annotations = []
        for id, (img_id, validation_score) in enumerate(zip(img_ids, validation_scores)):
            # validation_scores is a list of size two, where confidence is index = 1
            # element and caption is the index of the maximum confidence
            caption = 1 if validation_score[1] > validation_score[0] else 0
            confidence = validation_score[1] if caption == 1 else validation_score[0]
            caption_annotations.append(
                {
                    "id": id,
                    "image_id": img_id,
                    "category_id": 2,  # TODO: Currrently Hardcoded
                    "caption": caption,
                    "conf": confidence,
                }
            )

        return caption_annotations

    def _check_caption_annotation(self):
        """Checks if caption annotations has been returned by model"""
        return "validation_scores" in self.store_dict

    def _check_box_annotations(self):
        """Checks if box annotations have been returned by model"""
        return (
            ("boxes" in self.store_dict)
            and ("labels" in self.store_dict)
            and ("scores" in self.store_dict)
        )

    def resize_boxes(self, detections, image_shapes, original_image_shapes):
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(detections, image_shapes, original_image_shapes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            detections[i]["boxes"] = boxes

        return detections


class ObjectDetectionWithRejectWriter(ObjectDetectionWriter):
    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Helper function to be called at the end of each batch. This hook function append the
        predictions to the preds dictionary.

        Parameters
        ----------
        trainer : pl.Trainer
            pytorch lightning Trainer object
        pl_module : pl.LightningModule
            pytorch lightning Module object
        outputs : STEP_OUTPUT
            outputs of the model
        batch : Any
            batch of data
        batch_idx : int
            batch index
        dataloader_idx : int
            dataloader index
        """
        # Get detections
        detections = self.resize_boxes(
            outputs,
            image_shapes=len(outputs) * [[batch["img"].shape[-2], batch["img"].shape[-1]]],
            original_image_shapes=batch["img_size"],
        )
        boxes = [x["boxes"] for x in detections]
        labels = [x["labels"] for x in detections]
        scores = [x["scores"] for x in detections]
        validation_scores = [x["validation_scores"] for x in detections]

        # extend store_dict with information from current batch
        self.store_dict["boxes"].extend(boxes)
        self.store_dict["labels"].extend(labels)
        self.store_dict["scores"].extend(scores)
        self.store_dict["img_ids"].extend(batch["img_id"])
        self.store_dict["validation_scores"].extend(validation_scores)
