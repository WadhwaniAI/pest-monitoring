import os
from os.path import join
from typing import Optional

import pytorch_lightning as pl
import torch
from nni.compression.pytorch import ModelSpeedup
from omegaconf import OmegaConf as oc
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from compression.helper import OptimizerReload
from src.utils import utils

log = utils.get_logger(__name__)


class AbstractPruner:
    """Pytorch Lightning AbstractPruner Class"""

    def __init__(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        config_list: list,
        dummy_input_shape: Optional[tuple] = None,
        dependency_aware: bool = True,
        start_ckpt_path: str = None,
        gpus: int = 1,
        additional_pruner_args: dict = None,
    ):
        self.model = model
        self.config_list = config_list
        self.dummy_input = self._set_dummy_input(dummy_input_shape)
        self.dependency_aware = dependency_aware
        self.start_ckpt_path = start_ckpt_path
        self.gpus = gpus
        self.additional_pruner_args = (
            {} if additional_pruner_args is None else oc.to_container(additional_pruner_args)
        )

        # Setup dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()

    def setup(self):
        # Setup pruner
        self._set_pruner()

    def compress(self):
        self.model = self.pruner.compress()
        self.pruner.get_pruned_weights()

    def generate_masks(self):
        """Generate Masks"""
        temp_path = "post_finetuning"
        # Create temp directory
        os.makedirs(temp_path, exist_ok=True)
        network_path = join(temp_path, "model.pth")
        mask_path = join(temp_path, "mask.pth")
        self.pruner.export_model(model_path=network_path, mask_path=mask_path)
        return mask_path

    def speedup(self):
        """Using masks to decrease Model Size"""
        mask_path = self.generate_masks()
        self.pruner._unwrap_model()
        m_speedup = ModelSpeedup(self.model, self.dummy_input, mask_path)
        m_speedup.speedup_model()

    def run(self):
        """Runs the Pruning Process"""
        raise NotImplementedError

    def _set_pruner(self):
        raise NotImplementedError

    def _set_dummy_input(self, dummy_input_shape):
        """Set dummy input for pruning"""
        if dummy_input_shape is None:
            log.info("dummy_input_shape is not provided, using default of (1, 3, 512, 512)")
            return torch.randn(1, 3, 512, 512)
        return torch.randn(*dummy_input_shape)


class AbstractOneShotPruner(AbstractPruner):
    """Pytorch Lightning AbstractOneShotPruner Class"""

    def __init__(
        self,
        one_shot_finetuning_epochs: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.one_shot_finetuning_epochs = one_shot_finetuning_epochs

    def finetune(self):
        self.model.train()

        assert self.start_ckpt_path is not None, "start_ckpt_path is not provided"
        cb = OptimizerReload(self.start_ckpt_path)
        trainer = pl.Trainer(
            gpus=self.gpus,
            default_root_dir=None,
            logger=False,
            max_epochs=self.one_shot_finetuning_epochs,
            callbacks=[
                cb,
                self._set_early_stopping(),
                self._set_model_checkpoint(),
            ],
        )

        # fit the model
        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

    def run(self):
        self.setup()

        # Generating Masks based on Pruner
        log.info("Generating Masks")
        self.compress()

        # finetune the model
        log.info("Finetuning the model")
        self.finetune()

        # Perform Speedup using Masks
        log.info("Performing Speedup")
        self.speedup()

    def _set_model_checkpoint(self):
        return ModelCheckpoint(
            dirpath="checkpoints-fine-tuning",
            monitor="val/loss",
            mode="min",
            filename="{epoch:02d}",
        )

    def _set_early_stopping(self):
        return EarlyStopping(monitor="val/loss", mode="min", patience=5)


class AbstractIterativePruner(AbstractPruner):
    def __init__(
        self,
        pruning_algorithm: str = "l2",
        total_iteration: int = 10,
        keep_intermediate_result: bool = None,
        use_evaluator: bool = False,
        log_dir: str = ".",
        fine_tuning_epochs: int = 1,
        *args,
        **kwargs,
    ):
        """
        AGPPruner:
        For more details please refer to the Paper
        To prune, or not to prune: exploring the efficacy of pruning for model compression.
        [Paper](https://arxiv.org/abs/1710.01878)

        Parameters
        ----------
        pruning_algorithm : str
            The pruning algorithm to use. One of
            ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm']
        total_iteration : int
            The total number of iterations to run.
        keep_intermediate_result : bool
            Whether to keep the intermediate results (masks and weights) or not.
        speedup_model : bool
            Whether to speedup the model or not
        log_dir : str
            The directory to save the logs.
        fine_tuning_epochs : int
            The number of epochs to finetune the model.
        kwargs: dict
            Additional arguments to pass to the pruning algorithm.
        """
        super().__init__(*args, **kwargs)
        self.pruning_algorithm = pruning_algorithm
        self.total_iteration = total_iteration
        self.keep_intermediate_result = keep_intermediate_result
        self.use_evaluator = use_evaluator
        self.log_dir = log_dir
        self.fine_tuning_epochs = fine_tuning_epochs
        self.kwargs = kwargs

    def _set_task_generator(self):
        raise NotImplementedError

    def _set_pruning_algorithm(self):
        raise NotImplementedError

    def _set_pruning_scheduler(self, pruner, task_generator):
        raise NotImplementedError

    def _set_pruner(self):
        pruner = self._set_pruning_algorithm()

        task_generator = self._set_task_generator()

        self.scheduler = self._set_pruning_scheduler(pruner, task_generator)

    def _configure_trainer_evaluation(self):
        return pl.Trainer(
            gpus=self.gpus,
            default_root_dir=None,
            logger=False,
        )

    def _configure_trainer_finetuning(self, ckpt_path):
        return pl.Trainer(
            gpus=self.gpus,
            max_epochs=self.fine_tuning_epochs,
            default_root_dir=None,
            logger=False,
            num_sanity_val_steps=0,
            callbacks=[
                OptimizerReload(ckpt_path),
                self._set_model_checkpoint(),
                self._set_early_stopping(),
            ],
        )

    def _set_model_checkpoint(self):
        return ModelCheckpoint(
            dirpath="checkpoints-fine-tuning",
            monitor="val/loss",
            mode="min",
            filename="{epoch:02d}",
        )

    def _set_early_stopping(self):
        return EarlyStopping(monitor="val/loss", mode="min", patience=10)

    def compress(self):
        self.scheduler.compress()

    def speedup(self):
        _, model, masks, _, _ = self.scheduler.get_best_result()
        self.model = model
        ModelSpeedup(self.model, self.dummy_input, masks).speedup_model()

    def finetuner(self, model):
        log.info(f"Finetuning the model, starting optimizer state from {self.start_ckpt_path}")
        trainer = self._configure_trainer_finetuning(self.start_ckpt_path)

        # Fit the model for 1 epoch
        trainer.fit(model, self.train_dataloader, self.val_dataloader)
        self.start_ckpt_path = trainer.checkpoint_callback.best_model_path

    def evaluator(self, model):
        trainer = self._configure_trainer_evaluation()
        loss_dict = trainer.validate(model, self.val_dataloader)[0]
        return -loss_dict["val/loss"]  # As the evaluator tries to maximize the score

    def run(self):
        # Run the Setup
        self.setup()

        # Generating Masks based on Pruner
        log.info("Generating Masks")
        self.compress()

        # Speedup the model
        log.info("Speedup the model")
        self.speedup()
