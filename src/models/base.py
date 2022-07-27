from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection


class Model(pl.LightningModule):
    """Base Pytorch Lightning LightningModule Class

    Parameters
    ----------
    model_config: DictConfig
        Hydra config
    learning_rate: float
        Learning rate
    """

    def __init__(self, model_config: DictConfig, learning_rate: float = 0.01, **kwargs):
        """Initialize the model
        Set the model_config, learning_rate, network, criterion and metrics

        Parameters
        ----------
        learning_rate : float
            Learning rate for training
        model_config : DictConfig
            Config for the model
        """
        super().__init__()
        self.model_config = model_config
        self.learning_rate = learning_rate

        self.network: Module = hydra.utils.instantiate(self.model_config.network)
        self.criterion: Module = hydra.utils.instantiate(self.model_config.loss)

        metrics = MetricCollection(
            [hydra.utils.instantiate(metric) for metric in self.model_config.metrics.metric_list]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor):
        """Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            Input image

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        return self.network(x)

    def configure_optimizers(self):
        """Configure the optimizer
        If using a scheduler, the scheduler should be configured in this method.

        Returns
        -------
        Union[dict, Optimizer]
            If `lr_scheduler` is passed in `model_config`, then returns a dictionary with keys
            ["optimizer", "lr_scheduler"]
            Else returns the optimizer
        """
        optimizer: Optimizer = hydra.utils.instantiate(
            self.model_config.optimizer, lr=self.learning_rate, params=self.parameters()
        )
        if "lr_scheduler" in self.model_config:
            lr_scheduler = hydra.utils.instantiate(
                self.model_config.lr_scheduler, optimizer=optimizer
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def step(self, batch: Any):
        """Step function for the model

        Parameters
        ----------
        batch : Any
            Input batch

        Returns
        -------
        loss : LossValue
            Loss of the batch
        out : torch.Tensor
            Predicted output of the batch
        y : torch.Tensor
            Ground-truth of the batch
        """
        x, y = batch
        out = self.forward(x.float())
        loss = self.criterion(out, y)
        return loss, out, y

    def training_step(self, batch: Any, batch_idx: int):
        """Training step function for the model

        Parameters
        ----------
        batch : Any
            Input batch

        Returns
        -------
        LossValue
            Loss computed via the criterion
        """
        loss, out, y = self.step(batch)

        output = self.train_metrics(F.softmax(out, dim=1), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step function for the model

        Parameters
        ----------
        batch : Any
            Input batch
        """
        loss, out, y = self.step(batch)

        output = self.valid_metrics(F.softmax(out, dim=1), y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        """Test step function for the model

        Parameters
        ----------
        batch : Any
            Input batch
        """
        loss, out, y = self.step(batch)

        output = self.test_metrics(F.softmax(out, dim=1), y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output, on_step=False, on_epoch=True)
