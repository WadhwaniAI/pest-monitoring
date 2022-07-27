"""Compression Pipeline: Performs Pruning

To run the compression pipeline, run the following command:
python compress.py -cn config_path +ckpt='checkpoint_path' ++name=experiment_name

Arguments
---------
-cn (str): Path of the config file wrt the configs/ directory.
+ckpt (str): Absolute path to checkpoint file
++name (str): Name of the experiment (Compulsory to pass if not passing test_run=True)
+test_run (bool): If set to True, the name check will be skipped.

Additional Information
----------------------
+ : Appending key to a config
++ : Overriding key in a config
ckpt: If Checkpoint has an = in the path, then consider using `/=` instead of the
`=` OR wrap it like '+ckpt="checkpoint_path"'
"""

from typing import Optional

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything

from compression.helper import update_results
from src.utils import utils
from utils.check_config import check_compress_config

OUT_DIR = "/output/"
log = utils.get_logger(__name__)


def compress(config: DictConfig, ckpt_PATH: str) -> Optional[float]:
    """Contains training with Model Compression pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Parameters
    ----------
    config : DictConfig
        Configuration object
    ckpt_PATH : str
        Path to checkpoint file
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(
        config.datamodule, data_config=config.datamodule, _recursive_=False
    )
    datamodule.setup()

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}> and {ckpt_PATH}")
    model: LightningModule = instantiate(
        config.model, model_config=config["model"], _recursive_=False
    ).load_from_checkpoint(ckpt_PATH, model_config=config["model"], _recursive_=False)

    # Init Wandb Logger
    log.info("Instantiating Wandb Logger")
    wandb.init(
        name=config.name,
        config=config,
        project="pest-monitoring-new",
        notes="Compression Process: Reducing Model Size for Low Cost Inference",
    )

    # Evaluator Trainer
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir=None,
        logger=False,
    )

    # used to save the performance of the original & pruned & finetuned models
    dummy_input_shape = list(config.pruner.dummy_input_shape)
    results = update_results({}, "original", dummy_input_shape, model, trainer, datamodule)

    # Instantiate Pruner
    log.info("Started Pruning Process")
    pruner = instantiate(
        config.pruner, model=model, datamodule=datamodule, start_ckpt_path=ckpt_PATH
    )
    pruner.run()

    # Update model
    model = pruner.model

    # After Model Speedup, compute the flops, params and model performance
    results = update_results(results, "pruned", dummy_input_shape, model, trainer, datamodule)

    # Save the best model in jit mod
    model.eval()

    # Save compressed model as jit file
    trace = torch.jit.trace(model, torch.randn(dummy_input_shape))
    torch.jit.save(trace, "jit-checkpoint.pt")

    wandb.log(
        {
            "Compression Summary": wandb.Table(
                dataframe=pd.DataFrame.from_dict(results).reset_index()
            )
        }
    )

    # Make sure everything closed properly
    log.info("Finalizing!")


@hydra.main(config_path="../configs/")
def main(config: DictConfig):
    # Check if config has the right structure
    with open_dict(config):
        check_compress_config(config)

    return compress(config, config.ckpt)


if __name__ == "__main__":
    main()
