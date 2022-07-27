from functools import partial
from typing import Mapping, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from .base_collate import dict_collate, torchvision_dict_collate
from .transforms import Compose


class BaseDataModule(LightningDataModule):
    """BaseDataModule Class based on Pytorch Lightning LightningDataModule"""

    def __init__(
        self,
        data_config: DictConfig,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: Optional[Union[bool, Mapping[str, bool]]] = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        sampler_config: Optional[Union[DictConfig, Mapping[str, DictConfig]]] = None,
        **kwargs,
    ) -> None:
        """General custom torch lightning DataModule class for counting and
        validation tasks. It sets up and makes available dataloaders for all
        3 splits to lightning Trainer.

        Parameters
        ----------
        data_config : DictConfig
            Hydra Dictconfig
        num_workers : int, optional
            Pytorch Dataloader ``num_workers``, by default 16
        batch_size : int, optional
            Pytorch Dataloader ``batch_size``, by default 32
        shuffle : Optional[Union[bool, Mapping[str, bool]]], optional
            Pytorch Dataloader ``shuffle``, by default True
        drop_last : bool, optional
            Pytorch Dataloader ``drop_last``, by default True
        pin_memory : bool, optional
            Pytorch Dataloader ``pin_memory``, by default True
        sampler_config : Optional[Union[DictConfig, Mapping[str, DictConfig]]], optional
            Hydra DictConfig or Mapping[str, DictConfig] for sampler, by default None
        """
        super().__init__()

        self.data_config = data_config
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = None
        self.dataset_val = None
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.sampler_config = sampler_config
        self.collate_fn = self._setup_collate()

    def _setup_collate(self):
        """Set the collate function"""
        if "collate_fn" in self.data_config:
            if self.data_config.collate_fn == "torchvision_dict_collate":
                return partial(torchvision_dict_collate, include_background=True)
            elif self.data_config.collate_fn == "torchvision_dict_collate_without_background":
                return partial(torchvision_dict_collate, include_background=False)
            else:
                return NotImplementedError
        return dict_collate

    def _setup_transforms(self):
        """Instantiates a sequence of transforms to be applied to the datasets for
        train, val and test modes in self.transforms.
        """
        transforms_config = self.data_config.dataset.transforms
        keys = ["train", "val", "test"]
        self.transforms = {}
        for key in keys:
            if key in transforms_config:
                self.transforms[key] = Compose(
                    [instantiate(transform) for transform in transforms_config[key]]
                )
            else:
                raise ValueError(f"{key} is not in transforms_config")

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiates train and val dataset in ``self.dataset_train``
        and ``self.dataset_val`` respectively.

        Parameters
        ----------
        stage : Optional[str], optional
            Pytorch Lightning LightningDataModule ``stage``, defaults to None
        """
        self._setup_transforms()
        self.dataset_train: Dataset = instantiate(
            self.data_config.dataset,
            dataset_config=self.data_config.dataset,
            mode="train",
            transforms=self.transforms["train"],
            _recursive_=False,
        )
        self.dataset_val: Dataset = instantiate(
            self.data_config.dataset,
            dataset_config=self.data_config.dataset,
            mode="val",
            transforms=self.transforms["val"],
            _recursive_=False,
        )

    def _setup_sampler(self, mode: str, dataset: Dataset) -> Sampler:
        """Sets the sampler for the give split mode and dataset.
        If sampler_config is a DictConfig and it is not a mapping,
        instantiate it and return it. If sampler_config is a DictConfig and it
        is a Mapping[str, DictConfig], instantiate it for that mode.
        Else, return None

        Parameters
        ----------
        mode : str
            split mode, generally one of ["train", "val", "test"]
        dataset : Dataset
            Pytorch Dataset Class
        """
        if isinstance(self.sampler_config, DictConfig):
            if "_target_" in self.sampler_config:
                return instantiate(self.sampler_config, dataset=dataset)
            if mode in self.sampler_config:
                return instantiate(self.sampler_config[mode], dataset=dataset)

        return None

    def _setup_shuffle(self, mode: str) -> bool:
        """Sets the shuffle for the give split mode.
        If self.shuffle is a Mapping[str, bool] and mode is present in self.shuffle,
        returns value for the given mode. If mode is absent, returns False.
        If self.shuffle is a bool, returns self.shuffle

        Parameters
        ----------
        mode : str
            split mode, generally one of ["train", "val", "test"]
        """
        if isinstance(self.shuffle, DictConfig):
            if mode in self.shuffle:
                return self.shuffle[mode]
            else:
                return False
        return self.shuffle

    def train_dataloader(self) -> DataLoader:
        """Instantiate and return train dataloader

        Returns
        -------
        loader: DataLoader
            Pytorch Dataloader for train set
        """
        sampler = self._setup_sampler("train", self.dataset_train)
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False if sampler is not None else self._setup_shuffle("train"),
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Instantiate and return val dataloader

        Returns
        -------
        loader: DataLoader
            Pytorch Dataloader for val set
        """
        sampler = self._setup_sampler("val", self.dataset_val)
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False if sampler is not None else self._setup_shuffle("val"),
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Instantiates test dataset into ``self.dataset_test`` and
        test dataloader to return test dataloader.

        Returns
        -------
        loder: DataLoader
            Pytorch Dataloader for test set
        """
        self.dataset_test: Dataset = instantiate(
            self.data_config.dataset,
            dataset_config=self.data_config.dataset,
            mode="test",
            transforms=self.transforms["test"],
            _recursive_=False,
        )
        sampler = self._setup_sampler("test", self.dataset_test)
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False if sampler is not None else self._setup_shuffle("test"),
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        return loader
