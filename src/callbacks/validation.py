import getpass
import json
import os
from os.path import basename
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils import utils

from .base import AbstractSavePredictions

log = utils.get_logger(__name__)


class BinaryValSavePreds(AbstractSavePredictions):
    """BinaryValSavePreds Callback for saving predictions as JSON files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_store_dict(self):
        "Helps init store_dict."
        return {"img_ids": [], "category_ids": [], "preds": [], "confs": []}

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
        epoch = pl_module.current_epoch
        img_ids = self.store_dict["img_ids"]
        category_ids = self._convert_to_lists(self.store_dict["category_ids"])
        preds = self._convert_to_lists(self.store_dict["preds"])
        confs = self._convert_to_lists(self.store_dict["confs"])

        # Check if save dir exists else create it
        os.makedirs(self.save_dir, exist_ok=True)

        """Things we need to store: json

        So we have populate the following tables.
        info - stores the information about the run (needs to change)
            - version
            - description
            - contributor
            - url
            - date_created
        images - stores the images, will remain the same ✅
        box_annotations - stores the bounding boxes, will be empty for now
        caption_annotations - stores the captions, will have the following coloumns
            - id
            - image_id
            - category_id
            - caption (predicted-label)
            - conf
        categories - stores the categories, can remain the same ✅
            - id
            - name
            - supercategory
        splits - stores the splits, will remain the same ✅
        """
        # images
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

        # Save predictions to json file and fill info key of dict with information
        data_file = basename(self.data_file).split(".json")[0]
        pred_file = os.path.join(
            self.save_dir,
            f"{self.prefix}_{data_file}_{epoch}.json",
        )
        description = (
            f"Prediction of given model of data version {data_file} at"
            f" time {self.callback_time} on {self.date}. Prediction dump from `BinaryValSavePred`."
        )
        self.final_json["info"] = {
            "version": data_file,
            "description": description,
            "contributor": getpass.getuser(),
            "url": pred_file,
            "date_created": self.date,
        }

        # Make box annotations empty
        self.final_json["box_annotations"] = []

        # Make caption annotatiosn
        caption_annotations = []
        for id, (img_id, category_id, pred, conf) in enumerate(
            zip(img_ids, category_ids, preds, confs)
        ):
            # print ("img_id, category_id, pred, conf")
            # print (list(map(type, (img_id, category_id, pred, conf))))
            caption_annotations.append(
                {
                    "id": id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "caption": pred,
                    "conf": conf,
                }
            )
        self.final_json["caption_annotations"] = caption_annotations

        log.info(f"Saving predictions to {pred_file}")
        with open(pred_file, "w") as f:
            json.dump(self.final_json, f)

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
        This hook appends information to the store_dict.

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
        # getting preds
        logits = outputs["out"]
        category_ids = outputs["category_id"]
        softmax_vals = torch.softmax(logits, dim=1)
        assert softmax_vals.shape == logits.shape
        confs, preds = torch.max(softmax_vals, dim=1)
        assert preds.shape == (logits.shape[0],)
        assert confs.shape == preds.shape

        # extending store_dict with info from current batch
        self.store_dict["category_ids"].extend(category_ids)
        self.store_dict["preds"].extend(preds)
        self.store_dict["confs"].extend(confs)
        self.store_dict["img_ids"].extend(outputs["img_id"])
