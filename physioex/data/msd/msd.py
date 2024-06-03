from typing import Callable, Dict, List

import numpy as np

from physioex.data.base import PhysioExDataset

from physioex.data.dreem.dreem import Dreem
from physioex.data.mass.mass import Mass
from physioex.data.shhs.shhs import Shhs
from physioex.data.sleep_edf.sleep_edf import SleepEDF

datasets = {"sleep_physionet": SleepEDF, "dreem": Dreem, "shhs": Shhs, "mass": Mass}


class MultiSourceDomain(PhysioExDataset):
    def __init__(
        self,
        domains: List[Dict],
        preprocessing: str = "raw",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 1,
        target_transform: Callable = None,
        num_folds: int = 1,
    ):

        # assert each domain has a dataset key and a version key
        for domain in domains:
            assert "dataset" in domain, "dataset key is missing in domains argument"
            assert "version" in domain, "version key is missing in domains argument"
            assert "picks" in domain, "picks key is missing in domains argument"

        self.datasets = []
        self.offsets = [0]

        for domain in domains:
            self.datasets.append(
                datasets[domain["dataset"]](
                    version=domain["version"],
                    picks=domain["picks"],
                    preprocessing=preprocessing,
                    sequence_length=sequence_length,
                    target_transform=target_transform,
                )
            )

            self.offsets.append(self.datasets[-1].__len__() + self.offsets[-1])

        self.num_folds = num_folds

    def get_num_folds(self):
        return self.num_folds

    def split(self, fold: int = 0):
        for dataset in self.datasets:
            dts_num_folds = dataset.get_num_folds()

            selected_fold = np.random.randint(dts_num_folds)

            dataset.split(selected_fold)

    def __getitem__(self, idx):

        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= idx and idx < self.offsets[i + 1]:
                return self.datasets[i][idx - self.offsets[i]]

        raise IndexError(f"Wrong index {idx} on offsets {self.offsets}")

    def __len__(self):
        return self.offsets[-1]

    def get_sets(self):
        total_train_idx = []
        total_valid_idx = []
        total_test_idx = []

        for i, dataset in enumerate(self.datasets):
            train_idx, valid_idx, test_idx = dataset.get_sets()
            total_train_idx.append(train_idx + self.offsets[i])
            total_valid_idx.append(valid_idx + self.offsets[i])
            total_test_idx.append(test_idx + self.offsets[i])

        return (
            np.concatenate(total_train_idx),
            np.concatenate(total_valid_idx),
            np.concatenate(total_test_idx),
        )
