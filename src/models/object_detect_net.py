from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from .utils import detach


class ObjectDetectNet(pl.LightningModule):
    """Object Detection Base Model Class: Based on pl.LightningModule.
    Generic Lightning Module class for Object Detection"""

    def __init__(
        self,
        model_config: DictConfig,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize the model, set the model configuration, learning rate

        Parameters
        ----------
        model_config : DictConfig
            The model config containing information regarding network, optimizer,
            learning rate schedulers, etc.
        learning_rate : float
            learning rate of the optimizer, initialized here to enable
            auto_lr_find to work.
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.model_config = model_config

        self.network: Module = instantiate(self.model_config.network)

    def forward(self, images: torch.Tensor, targets: Optional[Any] = None) -> Any:
        """Forward pass of the model

        Parameters
        ----------
        images : torch.Tensor
            The images to be passed to the network post preprocessing.
        targets : Optional[Any]
            The targets to be passed to the network post preprocessing.
        """
        return self.network(images, targets)

    def predict(self, images: torch.Tensor) -> Any:
        """Calling the predict function from the network"""
        # if network does not have a predict function, raise error
        if not hasattr(self.network, "predict"):
            raise NotImplementedError(
                "Network does not have a predict function. Please implement one."
            )

        return self.network.predict(images)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Standard Pytorch Lightning function. Configures the optimizer
        If using a scheduler, the scheduler should be configured in this method.
        """
        optimizer: Optimizer = instantiate(
            self.model_config.optimizer, lr=self.learning_rate, params=self.parameters()
        )
        lr_scheduler = self._configure_lr_scheduler(optimizer)
        if lr_scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def _configure_lr_scheduler(self, optimizer):
        """Configure and returns the learning rate scheduler"""
        if "lr_scheduler" in self.model_config:
            return instantiate(self.model_config.lr_scheduler, optimizer=optimizer)
        return None

    def step(self, batch: Any) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Step function for the training, validation, and test phases
        This step consists of preprocessing the data, forward pass, and loss calculation.
        """
        images, glocs, glabels, img_ids, img_shapes = (
            batch["img"],
            batch["bbox_coord"],
            batch["bbox_class"],
            batch["img_id"],
            batch["img_size"],
        )

        # forward pass
        targets = self.pre_forward_step(glocs, glabels)
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

    def pre_forward_step(self, glocs, glabels):
        """Pre-forward step for target processing in the training, validation, and test phases.
        Does the following,
        - Replaces the nones in the glabels and glocs lists with empty tensors with correct device.
        - Converts the glocs and glabels to tensors with correct device.
        - Increments the glabels by 1 to account for the background class.
        - Converts to a list of dicts with the keys "boxes" and "labels".
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
        targets = [{"boxes": gloc, "labels": glabel.long()} for gloc, glabel in zip(glocs, glabels)]
        return targets

    def training_step(self, batch: Any, batch_idx: int):
        """Training step for the training phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "train")
        return step_output

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step for the validation phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "val")
        return step_output

    def test_step(self, batch: Any, batch_idx: int):
        """Test step for the test phase"""
        step_output = self.step(batch)

        self._log_loss(step_output, "test")
        return step_output

    def predict_step(self, batch: Any, batch_idx: int):
        """Test step for the test phase"""
        return self.predict(batch["img"])

    def _log_loss(self, step_output, prefix):
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


class SSDObjectDetectNet(ObjectDetectNet):
    """SSDObjectDetectNet Class for Deployment Prediction with
    torch.jit.export wrapper around predict()"""

    @torch.jit.export
    def predict(self, images: torch.Tensor):
        """Wrapper around predict_helper() for torch.jit.export"""
        return self.network.predict(images)
