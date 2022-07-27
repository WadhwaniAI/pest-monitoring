"""Evaluation Pipeline

To run the evaluation pipeline, run the following command:
python eval.py -cn config_path +ckpt='checkpoint_path' +split=val ++name=experiment_name

Arguments
---------
-cn (str): Path of the config file wrt the configs/ directory.
+ckpt (str): Absolute path to checkpoint file
+split (str): One of ["val", "test", "train"]
++name (str): Name of the experiment (Compulsory to pass if not passing test_run=True)
+test_run (bool): If set to True, the name check will be skipped.

Additional Information
----------------------
+ : Appending key to a config
++ : Overriding key in a config
ckpt: If Checkpoint has an = in the path, then consider using `/=` instead of the
`=` OR wrap it like '+ckpt="checkpoint_path"'
"""


import os
from typing import List

import dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, WandbLogger

from src.utils import utils
from utils.check_config import check_eval_config

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
OUT_DIR = "/output/"
log = utils.get_logger(__name__)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Function to get the wandb logger

    Parameters
    ----------
    trainer : Trainer
        pytorch lightning trainer object

    Returns
    -------
    WandbLogger
        pytorch lightning WandbLogger object
    """

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    return None


def evaluate(
    config: DictConfig,
    ckpt_PATH: str,
    split: str = "val",
):
    """Evaluate function to evaluate a model on a given split

    Parameters
    ----------
    config : DictConfig
        Configuration dictionary for setting up model, trainer, data, etc.
    ckpt_PATH : str
        Path to the checkpoint file
    split : str, optional
        Split to evaluate on, by default "val"
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(
        config.datamodule, data_config=config.datamodule, _recursive_=False
    )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = instantiate(
        config.model, model_config=config["model"], _recursive_=False
    ).load_from_checkpoint(ckpt_PATH, model_config=config["model"])

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(instantiate(lg_conf))

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf, _recursive_=False))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        _convert_="partial",
    )

    # Send parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Get Datamodule according to split
    log.info(f"Getting {split}_dataloader from datamodule")
    datamodule.setup()
    if split == "test":
        dataloader = datamodule.test_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:  # split == 'train'
        dataloader = datamodule.train_dataloader()

    # Perform Prediction using trainer.predict()
    log.info(f"Evaluating the model on {split}_dataloader")
    trainer.predict(model, dataloader)

    # Log data_file version from config on wandb
    data_file = config["datamodule"]["dataset"]["data_file"]
    logger = get_wandb_logger(trainer=trainer)
    if logger is not None:
        experiment = logger.experiment
        experiment.log(
            {
                "data_file": data_file,
                "ckpt_path": ckpt_PATH,
                "split": split,
            }
        )


@hydra.main(config_path="configs/")
def main(config: DictConfig):
    # Check if config has the right structure
    with open_dict(config):
        check_eval_config(config)

    return evaluate(config, config.ckpt, config.split)


if __name__ == "__main__":
    main()
