import os
from numbers import Integral
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pkg_resources as pkg
import yaml
from braindecode.datasets import SleepPhysionet as SP
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from braindecode.samplers import SequenceSampler
from loguru import logger
from sklearn.preprocessing import scale as standard_scale
from sklearn.utils import compute_class_weight

from physioex.data.base import PhysioExDataset
from physioex.data.utils import read_cache, write_cache


class SleepPhysionet(PhysioExDataset):
    def __init__(
        self,
        version: str = "2013",
        use_cache: bool = True,
        preprocessors: List[Preprocessor] = [
            Preprocessor(
                lambda data: np.multiply(data, 1e6), apply_on_array=True
            ),  # Convert from V to uV
            Preprocessor("filter", l_freq=None, h_freq=30),
        ],
        picks: str = "Fpz-Cz",
    ):
        assert version in ["2013", "2018"], "version should be one of '2013'-'2018'"
        # super(SleepPhysionet, self).___init__()
        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None
        self.window_dataset = None
        self.classes = None
        self.class_weights = None
        self.label_transform = get_center_label

        config = read_config(version)
        self.config = config

        cache_path = "temp/sleep_physionet_" + version + ".pkl"
        Path("temp/").mkdir(parents=True, exist_ok=True)

        if use_cache:
            self.windows_dataset = read_cache(cache_path)

        if self.windows_dataset:
            return

        logger.info("Fetching the dataset..")
        dataset = SP(
            subject_ids=config["subjects"], recording_ids=[1, 2], crop_wake_mins=30
        )

        # filtering
        preprocess(dataset, preprocessors)

        # windowing
        if picks is None:
            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=0,
                trial_stop_offset_samples=0,
                window_size_samples=config["window_size_s"] * config["sfreq"],
                window_stride_samples=config["window_size_s"] * config["sfreq"],
                preload=True,
                mapping=config["mapping"],
            )
        else:
            windows_dataset = create_windows_from_events(
                dataset,
                trial_start_offset_samples=0,
                trial_stop_offset_samples=0,
                window_size_samples=config["window_size_s"] * config["sfreq"],
                window_stride_samples=config["window_size_s"] * config["sfreq"],
                preload=True,
                mapping=config["mapping"],
                picks=picks,
            )

        # scaling
        preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

        self.windows_dataset = windows_dataset

        write_cache(cache_path, windows_dataset)

    def split(self, fold: int = 0):

        # train, valid, test split
        test_subject = fold

        config = self.config["splits"]
        splits = []
        desc = self.windows_dataset.description.copy()

        for subject in desc["subject"].values:
            if subject in config["test"][test_subject]:
                splits.append("test")
            elif subject in config["valid"][test_subject]:
                splits.append("valid")
            else:
                splits.append("train")
        df = pd.DataFrame([])
        df["split"] = splits

        self.windows_dataset.set_description(df, overwrite=True)

        splits = self.windows_dataset.split("split")
        train_set, valid_set, test_set = (
            splits["train"],
            splits["valid"],
            splits["test"],
        )

        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set

    def get_sets(self):
        return self.train_set, self.valid_set, self.test_set


def get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


@logger.catch
def read_config(version: str):
    if version == "2013":
        config_file = pkg.resource_filename(__name__, "config/sleep-edf-2013.yaml")
    elif version == "2018":
        config_file = pkg.resource_filename(__name__, "config/sleep-edf-2018.yaml")
    else:
        raise ValueError("bad version value in read_config: %s" % version)

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config
