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

from physioex.data.preprocessor import (Preprocessor, bandpass_filter,
                                        xsleepnet_preprocessing)
from physioex.data.sleep_edf.constant import (TOT_SLEEPEDF_NUM_WINDOWS,
                                              mapping, shape_raw,
                                              shape_xsleepnet, subjects)


class SLEEPEDFPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="mne_data",
            signal_shape=shape_raw,
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[shape_xsleepnet],
            data_folder=data_folder,
        )
        warnings.filterwarnings("ignore")

    @logger.catch
    def download_dataset(self) -> None:
        os.environ["MNE_DATA"] = self.dataset_folder
        sys.stdout = open(os.devnull, "w")
        SP(
            subject_ids=subjects,
            recording_ids=[1, 2],
            crop_wake_mins=30,
            load_eeg_only=False,
        )

        sys.stdout = sys.__stdout__

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        records = [str(record) for record in subjects]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

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

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        desc = pd.read_excel(
            "https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls"
        )

        desc = desc.drop(columns=["night", "LightsOff"]).drop_duplicates()

        table["sex"] = desc["sex (F=1)"]
        table["age"] = desc["age"]

        return table

    @logger.catch
    def get_sets(self) -> Tuple[List, List, List]:

        tmp_path = os.path.join(self.dataset_folder, "tmp.mat")
        urlretrieve(
            "https://github.com/pquochuy/xsleepnet/raw/master/sleepedf-78/data_split_eval.mat",
            tmp_path,
        )

        split_matrix = loadmat(tmp_path)

        train_subjects, valid_subjects, test_subjects = [], [], []
        for fold in range(len(split_matrix["test_sub"])):

            test_subjects.append(split_matrix["test_sub"][fold][0][0])
            valid_subjects.append(split_matrix["eval_sub"][fold][0][0])

            train_s = [
                subject
                for subject in subjects
                if subject not in test_subjects[-1]
                and subject not in valid_subjects[-1]
            ]
            train_subjects.append(train_s)

        os.remove(tmp_path)

        return train_subjects, valid_subjects, test_subjects

    @logger.catch
    def get_dataset_num_windows(self) -> int:

        return TOT_SLEEPEDF_NUM_WINDOWS


if __name__ == "__main__":

    p = SLEEPEDFPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
