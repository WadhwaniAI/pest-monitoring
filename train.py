"""Training pipeline

To run the training pipeline, run the following command:
python train.py -cn defaults/object-detection-ssd.yaml ++name=experiment_name

Arguments
---------
-cn (str): Path of the config file wrt the configs/ directory.
++name (str): Name of the experiment (Compulsory to pass if not passing test_run=True)
+test_run (bool): If set to True, the name check will be skipped.

Additional Information
----------------------
+ : Appending key to a config
++ : Overriding key in a config
"""

from typing import List, Optional

import dotenv
import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
from utils.check_config import check_train_config

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
OUT_DIR = "/output/"
log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Parameters
    ----------
    config : DictConfig
        Configuration object.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, data_config=config.datamodule, _recursive_=False
    )

    # Set learning rate from config.model.optimizer.lr, need to pass this incase
    # auto_lr_find is set to True
    if "auto_lr_find" not in config.trainer:
        log.info(f"Setting learning rate to {config.model.optimizer.lr}")
    learning_rate = config.model.optimizer.lr

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model, learning_rate=learning_rate, model_config=config["model"], _recursive_=False
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _recursive_=False))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # If Auto-lr is enabled, tune the model and update the learning rate
    if "auto_lr_find" in config.trainer:
        log.info("Auto-tuning the model")
        trainer.tune(model, datamodule)
        log.info("Auto-tuning finished")
        log.info("Updating learning rate")

        # Replace the learning rate passed to optimizer subdict in model config
        # with the new learning rate
        lr_fine_tuned = model.learning_rate
        log.info(f"Learning rate fine-tuned to {lr_fine_tuned}")
        log.info("Learning rate updated")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(logger=logger)


@hydra.main(config_path="configs/")
def main(config: DictConfig):
    # Check if config has the right structure
    with open_dict(config):
        check_train_config(config)

    return train(config)


if __name__ == "__main__":
    main()
