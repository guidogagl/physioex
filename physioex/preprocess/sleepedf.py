import os
import sys
import warnings
import zipfile
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from braindecode.datasets import SleepPhysionet as SP
from braindecode.preprocessing import Preprocessor as PR
from braindecode.preprocessing import create_windows_from_events, preprocess
from loguru import logger
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing

SLEEPEDF_SUBJECTS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    80,
    81,
    82,
]

mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


class SLEEPEDFPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):
        warnings.filterwarnings("ignore")

        super().__init__(
            dataset_name="sleepedf",
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        os.environ["MNE_DATA"] = self.dataset_folder
        sys.stdout = open(os.devnull, "w")
        SP(
            subject_ids=SLEEPEDF_SUBJECTS,
            recording_ids=[1, 2],
            crop_wake_mins=30,
            load_eeg_only=False,
        )

        sys.stdout = sys.__stdout__

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        records = [str(record) for record in SLEEPEDF_SUBJECTS]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        os.environ["MNE_DATA"] = self.dataset_folder

        # suppress printing
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        preprocessors = [
            PR(lambda data: np.multiply(data, 1e6), apply_on_array=True),
            PR("filter", l_freq=0.3, h_freq=40),
        ]

        dataset = SP(
            subject_ids=[int(record)],
            recording_ids=[1, 2],
            crop_wake_mins=30,
            load_eeg_only=False,
        )

        # filtering
        preprocess(dataset, preprocessors, n_jobs=-1)

        # windowing
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=30 * 100,
            window_stride_samples=30 * 100,
            preload=True,
            mapping=mapping,
            picks=["Fpz-Cz", "EOG horizontal", "EMG submental"],
            n_jobs=-1,
        )

        signals, labels = [], []
        for i in range(len(windows_dataset)):
            sig, label, _ = windows_dataset[i]

            signals.append(sig)
            labels.append(label)

        # restore printing
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        signals = np.array(signals)
        labels = np.array(labels)

        return signals, labels



if __name__ == "__main__":

    p = SLEEPEDFPreprocessor(data_folder="/mnt/vde/sleep-data/")

    p.run()
