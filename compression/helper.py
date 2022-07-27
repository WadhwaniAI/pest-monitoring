import time
from typing import Any, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nni.algorithms.compression.v2.pytorch.utils import OptimizerConstructHelper
from nni.compression.pytorch.utils.counter import count_flops_params
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.utilities.memory import get_model_size_mb
from tqdm import tqdm


class TaylorOptimizationLoop(OptimizerLoop):
    """TaylorOptimizationLoop using the nni traced optimizer"""

    def __init__(
        self, traced_optimizer: OptimizerConstructHelper, ckpt_path, device="cpu", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.optimizer = traced_optimizer
        self.ckpt_path = ckpt_path
        self.device = device
        self._load_optimizer_state_dict()

    def _load_optimizer_state_dict(self):
        try:
            checkpoint = torch.load(self.ckpt_path)
            optimizer_states = checkpoint["optimizer_states"][0]
            self.optimizer.load_state_dict(optimizer_states)
        except Exception as e:
            print(f"Error loading optimizer state dict: {e}")
            raise e

    def advance(self, batch: Any, *args: Any, **kwargs: Any):
        loss = self.trainer.lightning_module.step(batch)["loss"]

        # Manual Optimizeation Step
        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step() Not updating weights

        # Update progress
        self.optim_progress.optimizer_position += 1


class OptimizerReload(pl.Callback):
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def on_train_start(self, trainer, pl_module):
        ckpt = torch.load(self.ckpt_path)
        trainer.training_type_plugin.load_optimizer_state_dict(ckpt)


def get_model_time_cost(model, dummy_input, device=torch.device("cpu:0")):
    """Get model time cost

    Parameters
    ----------
    model : nn.Module
        model for which time needs to be calculated
    dummy_input : torch.Tensor
        dummy input for model
    device : torch.device
        device on which model is run
    """
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    n_times = 100
    time_list = []
    for _ in range(n_times):
        tic = time.time()
        try:
            _ = model(dummy_input)
        except Exception as e:
            raise f"Error in model: {e}"

        time_list.append(time.time() - tic)
    time_list = time_list[10:]
    return sum(time_list) / len(time_list)


def calibrate_model(model, loader, device=torch.device("cpu:0")):
    """Calibrate model

    Parameters
    ----------
    model : nn.Module
        model for which calibration needs to be done
    loader : torch.utils.data.DataLoader
        data loader for calibration
    device : torch.device
        device on which model is run
    """
    model.to(device)
    model.eval()

    for batch_id, batch in tqdm(enumerate(loader)):
        inputs = batch["img"].to(device)
        _ = model(inputs.float())


def get_pl_trainer(gpu=1):
    return pl.Trainer(
        gpus=gpu,
    )


def update_results(
    results,
    key: str,
    input_size: tuple,
    model: Union[pl.LightningModule, nn.Module],
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
):
    if key in results:
        raise ValueError(f"Key {key} already exists in results")

    results[key] = {}
    dummy_input = torch.randn(input_size)
    flops, params, _ = count_flops_params(model, dummy_input, verbose=False)
    results[key]["size (in Mb)"] = get_model_size_mb(model)
    results[key]["flops"] = flops
    results[key]["params"] = params
    results[key]["time (in sec)"] = get_model_time_cost(model, dummy_input)
    loss_dict = trainer.validate(model, datamodule.val_dataloader())
    results[key].update(loss_dict[0])
    return results
