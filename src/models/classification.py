from typing import Any

import torch.nn.functional as F

from src.loss.dtypes import BatchInfo

from .base import Model
from .utils import detach


class ClassificationModel(Model):
    """Base class for any machine learning model using Pytorch Lightning

    Parameters
    ----------
    config: Config
        Config Object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, batch: Any):
        """Step function for the model

        Parameters
        ----------
        batch : Any
            Input Batch

        Returns
        -------
        dict
            dictionary containing the keys ["loss", "out", "y", "category_id", "img_id"]
        """
        record = batch
        img = record["img"]
        y = record["label_value_cat"]
        category_id = record["label_class_cat"]
        out = self.forward(img)
        loss = self.criterion(BatchInfo(labels=out), BatchInfo(labels=y)).loss
        return {
            "loss": loss,
            "out": detach(out),
            "y": detach(y),
            "category_id": category_id,
            "img_id": record["img_id"],
        }

    def training_step(self, batch, batch_idx):
        """Training step function for the model

        Parameters
        ----------
        batch : Any
            Input batch

        Returns
        -------
        dict
            dictionary containing the keys ["loss", "out", "y", "category_id", "img_id"]
        """
        record = self._step(batch)
        loss, out, y = record["loss"], record["out"], record["y"]

        output = self.train_metrics(F.softmax(out, dim=1), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return record

    def validation_step(self, batch, batch_idx):
        """Validation step function for the model

        Parameters
        ----------
        batch : Any
            Input batch

        Returns
        -------
        dict
            dictionary containing the keys ["loss", "out", "y", "category_id", "img_id"]
        """

        record = self._step(batch)
        loss, out, y = record["loss"], record["out"], record["y"]

        output = self.valid_metrics(F.softmax(out, dim=1), y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True)
        return record

    def test_step(self, batch, batch_idx):
        """Test step function for the model

        Parameters
        ----------
        batch : Any
            Input batch

        Returns
        -------
        dict
            dictionary containing the keys ["loss", "out", "y", "category_id", "img_id"]
        """
        record = self._step(batch)
        loss, out, y = record["loss"], record["out"], record["y"]

        output = self.test_metrics(F.softmax(out, dim=1), y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True)
        return record
