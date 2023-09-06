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

class MultipleBiasedShuffleInfo(ShuffleInfoBase):
    base_bounds: tuple[int, int]
    extra_bounds: list[tuple[int, int]]

    def __init__(self, base_shuffle_info: list[dict], extra_shuffle_info: list[list[dict]],
                 extra_ratio: float=1.0, within_extra_ratios: list[float]=None):
        if within_extra_ratios is None:
            within_extra_ratios = np.ones(len(extra_shuffle_info), dtype=np.float64) / len(extra_shuffle_info)
        else:
            assert len(within_extra_ratios) == len(extra_shuffle_info)
            # normalize to probabilities
            within_extra_ratios = np.array(within_extra_ratios, dtype=np.float64) / np.sum(within_extra_ratios)

        self.extra_ratio = extra_ratio
        self.within_extra_ratios = within_extra_ratios

        self.base_bounds = (0, len(base_shuffle_info))
        self.extra_bounds = []

        self.shuffle_info_union = np.empty(len(base_shuffle_info) +
                                             np.sum([len(extra) for extra in extra_shuffle_info]), dtype=object)
        for i, info in enumerate(base_shuffle_info):
            self.shuffle_info_union[i] = info

        current_pos = len(base_shuffle_info)
        for extra in extra_shuffle_info:
            self.extra_bounds.append((current_pos, current_pos + len(extra)))
            for i, info in enumerate(extra):
                self.shuffle_info_union[current_pos + i] = info
            current_pos += len(extra)

        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.shuffle_info_union)

    def __getitem__(self, index):
        return self.shuffle_info_union[index]

    def get_random_shuffle_indices(self) -> np.ndarray:
        base_indices = np.arange(self.base_bounds[0], self.base_bounds[1], dtype=np.int32)
        base_size = len(base_indices)

        extra_indices = []
        for i, bounds in enumerate(self.extra_bounds):
            size_i = int(np.round(self.extra_ratio * self.within_extra_ratios[i] * base_size))
            while size_i > 0:
                if size_i < bounds[1] - bounds[0]:
                    extra_indices.append(self.rng.choice(bounds[1] - bounds[0], size=size_i, replace=False) + bounds[0])
                else:
                    extra_indices.append(np.arange(bounds[0], bounds[1], dtype=np.int32))
                size_i -= len(extra_indices[-1])

        all_indices = np.concatenate([base_indices] + extra_indices)
        shuffled_indices = self.rng.permutation(all_indices)
        return shuffled_indices
