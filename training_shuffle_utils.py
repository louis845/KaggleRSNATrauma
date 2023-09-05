import numpy as np
import abc

class ShuffleInfoBase(abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def get_random_shuffle_indices(self, *args) -> np.ndarray:
        pass

class ShuffleInfo(ShuffleInfoBase):
    def __init__(self, shuffle_info: list[dict]):
        self.shuffle_info = np.empty(len(shuffle_info), dtype=object)
        for i, info in enumerate(shuffle_info):
            self.shuffle_info[i] = info

        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.shuffle_info)

    def __getitem__(self, index):
        return self.shuffle_info[index]

    def get_random_shuffle_indices(self, size: int=None) -> np.ndarray:
        if size is None:
            return self.rng.permutation(len(self.shuffle_info))
        return self.rng.choice(len(self.shuffle_info), size=size, replace=False)

class BiasedShuffleInfo(ShuffleInfoBase):
    def __init__(self, shuffle_info: list[dict], shuffle_info_extra: list[dict], extra_shuffle_samples: int):
        assert extra_shuffle_samples <= len(shuffle_info_extra), "extra_shuffle_samples must be less than or equal to the number of extra samples"

        self.shuffle_info_union = np.empty(len(shuffle_info) + len(shuffle_info_extra), dtype=object)
        self.extra_shuffle_cutoff = len(shuffle_info)
        for i, info in enumerate(shuffle_info):
            self.shuffle_info_union[i] = info
        for i, info in enumerate(shuffle_info_extra):
            self.shuffle_info_union[i + self.extra_shuffle_cutoff] = info

        self.extra_shuffle_samples = extra_shuffle_samples
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.shuffle_info_union)

    def __getitem__(self, index):
        return self.shuffle_info_union[index]

    def get_random_shuffle_indices(self, size: int=None, extra_size: int=None) -> np.ndarray:
        if size is None:
            size = self.extra_shuffle_cutoff
        if extra_size is None:
            extra_size = self.extra_shuffle_samples

        indices = self.rng.choice(self.extra_shuffle_cutoff, size=size, replace=False) # indices for the first part
        extra_indices = self.rng.choice(len(self.shuffle_info_union) - self.extra_shuffle_cutoff,
                                        size=extra_size, replace=False) + self.extra_shuffle_cutoff # indices for the second part
        indices = np.concatenate((indices, extra_indices))
        indices = self.rng.permutation(indices)
        return indices
