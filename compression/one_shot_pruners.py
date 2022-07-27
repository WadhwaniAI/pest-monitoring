"""One-Shot Pruners"""
from nni.algorithms.compression.pytorch.pruning import (
    L1FilterPruner,
    L2FilterPruner,
    LevelPruner,
)
from omegaconf import OmegaConf as oc

from compression.pruner import AbstractOneShotPruner


class LevelPrunerOneShot(AbstractOneShotPruner):
    def _set_pruner(self):
        self.model.eval()
        self.pruner = LevelPruner(
            self.model,
            oc.to_container(self.config_list),
            dependency_aware=self.dependency_aware,
            dummy_input=self.dummy_input,
        )


class L1FilterPrunerOneShot(AbstractOneShotPruner):
    def _set_pruner(self):
        self.model.eval()
        self.pruner = L1FilterPruner(
            self.model,
            oc.to_container(self.config_list),
            dependency_aware=self.dependency_aware,
            dummy_input=self.dummy_input,
        )


class L2FilterPrunerOneShot(AbstractOneShotPruner):
    def _set_pruner(self):
        self.model.eval()
        self.pruner = L2FilterPruner(
            self.model,
            oc.to_container(self.config_list),
            dependency_aware=self.dependency_aware,
            dummy_input=self.dummy_input,
        )
