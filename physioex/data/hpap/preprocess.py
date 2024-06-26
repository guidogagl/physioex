import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pyedflib
from loguru import logger
from scipy.signal import resample
from tqdm import tqdm

from physioex.data.constant import get_data_folder
from physioex.data.preprocessor import (
    Preprocessor,
    bandpass_filter,
    xsleepnet_preprocessing,
)

url = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/datasets/homepap-baseline-harmonized-dataset-0.2.0.csv"

mapping = {
    0: 0,  # Wake
    1: 1,  # Stage 1
    2: 2,  # Stage 2
    3: 3,  # Stage 3
    4: 3,  # Stage 4
    5: 4,  # REM
    6: 5,
    9: -1,  # Not scored
}

eeg_labels = [("c3", "m2"), ("c3", "a2"), "c3-m2", ("c3", "o2")]
emg_labels = [
    ("lchin", "cchin"),
    ("lchin", "rchin"),
    ("chin1", "chin2"),
    "chin",
    "chin emg",
    "emg chin",
    "chin1-chin2",
    "lchin-cchin",
    ("emg3", "emg1"),
    ("emg3", "emg2"),
    "emg3-emg1",
    "emg3-emg2",
    "emg1-emg3",
    "emg",
    ("l-legs", "r-legs"),
]
ecg_labels = [
    ("ecg3", "ecg1"),
    ("ekg3", "ekg1"),
    ("ecg2", "ecg1"),
    ("ekg2", "ekg1"),
    "ecg3-ecg1",
    "ecg3-ecg2",
    "ecg1-ecg3",
    "ekg3-ekg2",
    "ekg",
    "ekg3-ekg1",
    "ecg",
    ("ecg ii", "ecg i"),
]
eog_labels = [
    ("e1", "e2"),
    ("e1-m2", "e2-m1"),
    ("loc", "roc"),
    ("e-1", "e-2"),
    ("l-eog", "r-eog"),
    "e1-e2",
    "e2-m1",
]


def extract_signal(f, possible_labels):
    signal_labels = [
        label.lower() for label in f.getSignalLabels()
    ]  # Converti tutte le etichette in minuscolo

    for label in possible_labels:
        # check if label is a tuple
        if isinstance(label, tuple):
            # check if the tuple is of 2 elements or 3
            if len(label) == 2:
                try:
                    signal = f.readSignal(signal_labels.index(label[0])) - f.readSignal(
                        signal_labels.index(label[1])
                    )
                    return signal
                except:
                    continue
            elif len(label) == 3:
                try:

                    Lchin = f.readSignal(
                        signal_labels.index(label[0])
                    )  # Cerca l'indice in modo case-insensitive
                    Rchin = f.readSignal(signal_labels.index(label[1]))
                    Cchin = f.readSignal(signal_labels.index(label[2]))
                    return np.sqrt(Lchin**2 + Rchin**2 + Cchin**2)
                except:
                    continue
        else:
            try:
                signal = f.readSignal(signal_labels.index(label))
                return signal
            except:
                continue

    logger.error(f"Labels {possible_labels} not found in : {signal_labels}")
    exit(-1)


def extract_fs(f):
    signal_labels = [
        label.lower() for label in f.getSignalLabels()
    ]  # Converti tutte le etichette in minuscolo

    for label in eeg_labels:
        # check if label is a tuple
        if isinstance(label, tuple):
            try:
                fs = int(f.getSampleFrequencies()[signal_labels.index(label[0])])
                return fs
            except:
                continue
        else:
            try:
                fs = int(f.getSampleFrequencies()[signal_labels.index(label)])
                return fs
            except:
                continue

    logger.error(f"Labels {eeg_labels} not found in : {signal_labels}")
    exit(-1)


class HPAPPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="hpap",
            signal_shape=[4, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[4, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # this method should be provided by the user
        # the method should return a list containing the path of each subject record
        # each path is needed to be passed as argument to the function read_subject_record(self, record)

        records = pd.read_csv(url)["nsrrid"].values
        records = [str(record) for record in records]

        list_files = os.listdir(os.path.join(self.dataset_folder, "download"))
        list_files = [
            file
            for file in list_files
            if file.endswith(".edf") and file.startswith("lab")
        ]

        result = []

        for record in records:

            files = [file for file in list_files if file.endswith(record + ".edf")]

            record_name = ""
            for file in files:
                record_name += file + " "
            if len(files) > 0:
                result.append(record_name)

        return result

    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        records = record.split(" ")
        download_folder = os.path.join(self.dataset_folder, "download")

        signal_records = []
        labels_records = []

        for my_record in records:
            if not my_record.endswith(".edf"):
                continue

            filepath = os.path.join(download_folder, my_record)
            anntpath = filepath.replace(".edf", "-profusion.xml")

            # logger.info(f"Reading xml record {my_record}")
            annt = ET.parse(anntpath).findall("SleepStages")[0]

            labels = np.zeros(len(annt))

            for i, annotation in enumerate(annt):
                labels[i] = int(annotation.text)

            # map the labels, remove the not scored windows
            labels = np.array([mapping[int(label)] for label in labels])
            labels = labels[labels != -1]
            labels = labels[labels != 5]
            return None, labels

            # logger.info(f"Reading edf record {my_record}")
            f = pyedflib.EdfReader(filepath)

            fs = extract_fs(f)
            EEG = extract_signal(f, eeg_labels)
            EOG = extract_signal(f, eog_labels)
            EMG = extract_signal(f, emg_labels)
            ECG = extract_signal(f, ecg_labels)

            f.close()

            # logger.info( f"Resampling and filtering record {my_record}")

            signal_record = np.array([EEG, EOG, EMG, ECG])

            signal_record = np.reshape(signal_record, (4, -1, fs))

            num_windows = signal_record.shape[1] // 30

            if num_windows != labels.shape[0]:
                logger.error(
                    f"Record {my_record} has {num_windows} windows and {labels.shape[0]} labels"
                )
                num_windows = min(num_windows, labels.shape[0])

            signal_record = signal_record[:, : num_windows * 30]
            labels = labels[:num_windows]

            signal_record = np.reshape(signal_record, (4, num_windows, 30 * fs))

            signal_record = resample(signal_record, num=30 * 100, axis=2)
            signal_record = bandpass_filter(signal_record, 0.3, 40, 100)

            signal_record = np.transpose(signal_record, (1, 0, 2))

            # remove the windows correspondig to movements
            mask = labels == 5
            signal_record = signal_record[~mask]
            labels = labels[~mask]

            signal_records.append(signal_record)
            labels_records.append(labels)

        # logger.info(f"Concatenating records")

        signal_records = np.concatenate(signal_records, axis=0)
        labels_records = np.concatenate(labels_records, axis=0)
        # logger.info(f"Records concatenated")

        return signal_records, labels_records

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        records = pd.read_csv(url)  # ["nsrrid", "nsrr_age", "nsrr_sex"]

        # map the nsrrsex column in 1:female and 0:male
        records["nsrr_sex"] = records["nsrr_sex"].map({"female": 1, "male": 0})

        list_files = os.listdir(os.path.join(self.dataset_folder, "download"))
        list_files = [
            file
            for file in list_files
            if file.endswith(".edf") and file.startswith("lab")
        ]

        selected_id = []
        for record in records["nsrrid"]:
            files = [file for file in list_files if str(record) in file]
            if len(files) > 0:
                selected_id.append(record)

        # select the records
        records = records[records["nsrrid"].isin(selected_id)]

        if len(records) != len(table):
            logger.error(
                f"Table has {len(table)} records and the records has {len(records)} records"
            )

        table["nsrrid"] = records["nsrrid"].values
        table["age"] = records["nsrr_age"].values
        table["sex"] = records["nsrr_sex"].values

        return table

    @logger.catch
    def get_sets(self) -> Tuple[np.array, np.array, np.array]:

        np.random.seed(42)

        table = self.table.copy()

        train_subjects = np.random.choice(
            table["subject_id"], size=int(table.shape[0] * 0.7), replace=False
        )
        valid_subjects = np.setdiff1d(
            table["subject_id"], train_subjects, assume_unique=True
        )
        test_subjects = np.random.choice(
            valid_subjects, size=int(table.shape[0] * 0.15), replace=False
        )
        valid_subjects = np.setdiff1d(valid_subjects, test_subjects, assume_unique=True)

        return (
            train_subjects.reshape(1, -1),
            valid_subjects.reshape(1, -1),
            test_subjects.reshape(1, -1),
        )

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        return 231434


if __name__ == "__main__":

    p = HPAPPreprocessor(data_folder="/home/guido/shared/")

    p.run()
