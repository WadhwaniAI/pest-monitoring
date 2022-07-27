"""Class to use for any rejection_network, which is a network
that does both object detection and classification"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .utils import detach


class CFNet(pl.LightningModule):
    """Base Pytorch Lightning Module Class for CFNet Training"""

    def __init__(
        self,
        model_config: DictConfig,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize the model, set the model configuration, learning rate, dboxes, box_encoder,
        criterion, and network

        Parameters
        ----------
        model_config : DictConfig
            Model configuration
        learning_rate : float, optional
            Learning rate, by default 0.01
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.model_config = model_config

        self.network: Module = instantiate(self.model_config.network)

    def forward(self, images: torch.Tensor, targets: Optional[Any] = None) -> Union[Dict, Tuple]:
        """Performs forward pass of the model and returns loss_dict if targets are passed, or
        output tensors as a tuple otherwise"""
        return self.network(images, targets)

    def predict(self, images: torch.Tensor) -> List[Dict[str, Tensor]]:
        """BoundingBox + Image Level Predictions"""
        return self.network.predict(images)

    def configure_optimizers(self) -> Dict:
        """Configure the optimizer
        If using a scheduler, the scheduler should be configured in this method.
        """
        optimizer: Optimizer = instantiate(
            self.model_config.optimizer, lr=self.learning_rate, params=self.parameters()
        )
        lr_scheduler = self._configure_lr_scheduler(optimizer)
        if lr_scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def _configure_lr_scheduler(self, optimizer) -> Any:
        """Configure and returns the learning rate scheduler"""
        if "lr_scheduler" in self.model_config:
            return instantiate(self.model_config.lr_scheduler, optimizer=optimizer)
        return None

    def step(self, batch: Any) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Step function for the training, validation, and test phases
        This step consists of preprocessing the data, forward pass, and loss calculation.
        """
        images, glocs, glabels, val_labels, img_ids, img_shapes = (
            batch["img"],
            batch["bbox_coord"],
            batch["bbox_class"],
            batch["label_value"],
            batch["img_id"],
            batch["img_size"],
        )

        # forward pass
        targets = self.pre_forward_step(glocs, glabels, val_labels)
        loss_dict = self(images, targets)

        step_output = {
            "images": detach(images),
            "targets": detach(targets),
            "img_ids": img_ids,
            "img_shapes": img_shapes,
        }
        step_output.update({k: detach(v) for k, v in loss_dict.items()})
        step_output.update({"loss": sum(loss_dict.values())})
        return step_output

    def pre_forward_step(
        self, glocs: Iterable, glabels: Iterable, val_labels: Iterable
    ) -> List[Dict]:
        """Pre-forward step for the training, validation, and test phases
        Replaces the nones in the glabels and glocs lists with empty tensors with correct device.
        """
        # Replace the nones in the glabels and glocs lists with empty tensors with correct device
        glocs = [
            torch.empty(0, 4, dtype=torch.float32, device=self.device) if gloc is None else gloc
            for gloc in glocs
        ]
        glabels = [
            torch.empty(0, dtype=torch.int64, device=self.device) if glab is None else glab
            for glab in glabels
        ]
        # labels should start from 1, 0 is background
        glabels = [glabel + 1 for glabel in glabels]
        targets = [
            {"boxes": gloc, "labels": glabel.long(), "image_label": val_label}
            for gloc, glabel, val_label in zip(glocs, glabels, val_labels)
        ]
        return targets

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Training step for the training phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "train")
        return step_output

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Validation step for the validation phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "val")
        return step_output

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Test step for the test phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "test")
        return step_output

    def predict_step(self, batch: Any, batch_idx: int) -> List[Dict[str, Tensor]]:
        """Predict step for the predict phase"""
        return self.predict(batch["img"])

    def _log_loss(self, step_output: Dict, prefix: str) -> None:
        """Log the loss for the training, validation, and test phases"""
        for (k, v) in step_output.items():
            is_loss = "loss" in k
            if is_loss:
                self.log(
                    f"{prefix}/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=is_loss,
                    logger=True,
                )


class CFSSDNet(CFNet):
    @torch.jit.export
    def predict(self, images: Tensor) -> List[Dict[str, Tensor]]:
        """Performs the NMS and returns the detections.

        Parameters
        ----------
        images : Tensor
            The input images tensor.

        Returns
        -------
        List[Dict[str, Tensor]]
            List with model's output for each image
        """
        return self.network.predict(images)
