"""Sampler Options"""

import random

import numpy as np
from torch.utils.data.sampler import Sampler


class CFSampler(Sampler):
    """CFSampler
    Decides the samples their ordering within an epoch.
    valid indices: images with objects
    invalid indices: images without objects
    If the dataset has no invalid indices, then it defaults to the
    functionality of the torch DataLoader.
    If the dataset has invalid indices, then the functionality depends
    on the value of `shuffle`.
    - If shuffle=True, then it samples invalid indices for each epoch
    from the set of all invalid indices such that the number of valid and invalid
    indices in the epoch are same. Also, within each batch, the number of valid and
    invalid indices are kept same.
    - If shuffle=False, then all the invalid indices are used for an epoch. In this
    case, each batch contains an equal number of valid and invalid indices until the
    number of valid indices are exhausted. After that, each batch only has invalid
    indices.
    Args:
        dataset: torch.utils.data.Dataset, the dataset object from which to sample
        shuffle: bool, decides the functionality for the sampler, default=True
        seed: int, random seed to use for sampling, default=0
    """

    def __init__(self, dataset, shuffle=True, seed=0):
        super(CFSampler, self).__init__(dataset)
        self.dataset = dataset
        self.valid_indices = dataset.valid_indices
        self.invalid_indices = dataset.invalid_indices
        self.num_valid = len(self.valid_indices)
        self.shuffle = shuffle
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        if len(self.invalid_indices) == 0:
            # set default behaviour if the dataset has
            # no invalid indices
            self.load_all_indices = self.load_default
            self.len = len(self.valid_indices)
        else:
            # set custom behaviours if dataset has invalid indices
            self.load_all_indices = self.load_combined

            if shuffle:
                # number of valid and invalid indices should be the
                # same for each epoch
                self.len = 2 * min(self.num_valid, len(self.invalid_indices))
            else:
                # use the complete dataset
                self.len = len(dataset)

    def load_default(self):
        """Default behaviour at torch DataLoader"""
        if self.shuffle:
            random.shuffle(self.valid_indices)
        return self.valid_indices

    def load_combined(self):
        """Custom behaviour for sampling indices based on self.shuffle"""
        if self.shuffle:
            random.shuffle(self.valid_indices)
            # sample invalid indices for the epoch
            epoch_invalid_indices = np.random.choice(
                self.invalid_indices, min(self.num_valid, len(self.invalid_indices)), replace=False
            ).tolist()

        else:
            # use all invalid indices for the epoch
            epoch_invalid_indices = self.invalid_indices

        if self.shuffle:
            # ensure each batch has equal number of valid and invalid indices
            indices = []
            for i in range(min(self.num_valid, len(epoch_invalid_indices))):
                indices.append(self.valid_indices[i])
                indices.append(epoch_invalid_indices[i])
        else:
            indices = list(range(len(self.dataset)))

        return indices

    def __iter__(self):
        return iter(self.load_all_indices())

    def __len__(self):
        return self.len
