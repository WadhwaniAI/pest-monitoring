"""Iterative Pruner Class"""

import nni
import pytorch_lightning as pl
import torch
from nni.algorithms.compression.v2.pytorch.pruning.basic_scheduler import (
    PruningScheduler,
)
from nni.algorithms.compression.v2.pytorch.pruning.tools import AGPTaskGenerator
from nni.algorithms.compression.v2.pytorch.utils import OptimizerConstructHelper
from nni.compression.pytorch.pruning import (
    L1NormPruner,
    L2NormPruner,
    TaylorFOWeightPruner,
)
from omegaconf import OmegaConf as oc

from compression.helper import TaylorOptimizationLoop
from compression.pruner import AbstractIterativePruner
from src.utils import utils

log = utils.get_logger(__name__)


class AGPIterativePruner(AbstractIterativePruner):
    def _set_task_generator(self):
        log.info("Setting AGPTaskGenerator for Scheduler")
        return AGPTaskGenerator(
            self.total_iteration,
            self.model,
            oc.to_container(self.config_list),
            log_dir=self.log_dir,
            keep_intermediate_result=self.keep_intermediate_result,
        )

    def _set_pruning_algorithm(self):
        log.info("Setting Pruning Algorithm for Scheduler")
        if self.pruning_algorithm == "l1":
            return L1NormPruner(None, None)
        elif self.pruning_algorithm == "l2":
            return L2NormPruner(None, None)
        elif self.pruning_algorithm == "taylorfo":
            training_batches = self.additional_pruner_args.get("training_batches", 20)
            return TaylorFOWeightPruner(
                None,
                None,
                training_batches=training_batches,
                criterion=None,
                traced_optimizer=self._get_traced_optimizer(),
                trainer=self.taylorfo_trainer,
            )
        else:
            raise ValueError(f"Pruning Algorithm {self.pruning_algorithm} is not supported")

    def _set_pruning_scheduler(self, pruner, task_generator):
        log.info("Setting Pruning Scheduler")
        return PruningScheduler(
            pruner,
            task_generator,
            finetuner=self.finetuner,
            speedup=False,
            dummy_input=self.dummy_input,
            evaluator=self.evaluator if self.use_evaluator else None,
            reset_weight=False,
        )

    def taylorfo_trainer(self, model, optimizer, criterion):
        """Helper trainer to be used with TaylorFOWeightPruner"""
        training_batches = self.additional_pruner_args.get("training_batches", 20)
        trainer = pl.Trainer(
            gpus=0,  # TODO: Support GPU Use
            max_epochs=1,
            default_root_dir=None,
            logger=False,
            limit_train_batches=training_batches,
            num_sanity_val_steps=0,
        )
        log.info(f"Running Taylor Optimization with training batch(es) : {training_batches}")
        trainer.fit_loop.epoch_loop.batch_loop.connect(
            optimizer_loop=TaylorOptimizationLoop(optimizer, self.start_ckpt_path)
        )
        trainer.fit(model, self.train_dataloader)

    def _get_traced_optimizer(self):
        dict_without_target = oc.to_container(self.model.model_config.optimizer.copy())
        del dict_without_target["_target_"]
        return OptimizerConstructHelper.from_trace(
            self.model,
            nni.trace(eval(self.model.model_config.optimizer["_target_"]))(
                params=self.model.parameters(), **dict_without_target
            ),
        )
