import os
import zipfile
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import pyedflib
import requests
from loguru import logger
from scipy.signal import butter, filtfilt, resample, spectrogram
from scipy.stats import mode
from tqdm import tqdm

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import bandpass_filter, xsleepnet_preprocessing


def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def extract_large_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            extracted_path = zf.extract(member, extract_path)
            if member.create_system == 3:  # If the OS of creating system is Unix
                unix_attributes = member.external_attr >> 16
                if unix_attributes:
                    os.chmod(extracted_path, unix_attributes)
    os.remove(zip_path)


SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]


def read_edf(file_path):

    f = pyedflib.EdfReader(os.path.join(file_path, "psg.edf"))
    labels = f.getSignalLabels()
    fs = 256

    idx_eeg = labels.index("C3-M2" if "C3-M2" in labels else "C4-M1")
    EEG = f.readSignal(idx_eeg).reshape(-1, fs)

    # Get the indices of the EOG signals
    idx_eog1 = labels.index("E1-M2")
    idx_eog2 = labels.index("E2-M2")

    # Read the EOG signals
    signal_eog1 = f.readSignal(idx_eog1).reshape(-1, fs)
    signal_eog2 = f.readSignal(idx_eog2).reshape(-1, fs)

    # Combine the EOG signals
    EOG = signal_eog1 + signal_eog2

    idx_emg = labels.index("CHIN")
    idx_ecg = labels.index("ECG-II")

    EMG = f.readSignal(idx_emg).reshape(-1, fs)
    ECG = f.readSignal(idx_ecg).reshape(-1, fs)

    f._close()
    # EEG shape n_seconds, fs
    n_windows = EEG.shape[0] // 30

    EEG, EOG, EMG, ECG = (
        EEG[: n_windows * 30].reshape(n_windows, 30 * fs),
        EOG[: n_windows * 30].reshape(n_windows, 30 * fs),
        EMG[: n_windows * 30].reshape(n_windows, 30 * fs),
        ECG[: n_windows * 30].reshape(n_windows, 30 * fs),
    )

    signal = np.transpose(np.array([EEG, EOG, EMG, ECG]), (1, 0, 2))

    # pass band the signal between 0.3 and 40 Hz
    signal = bandpass_filter(signal, 0.3, 40, 100)

    signal = resample(signal, num=30 * 100, axis=2)

    # Read the file
    hyp = pd.read_csv(os.path.join(file_path, "hypnogram.ids"), header=None).values

    start_index, end_index, hyp = (
        hyp[:, 0].astype(int),
        hyp[:, 1].astype(int),
        hyp[:, 2],
    )
    hyp = np.array([SLEEP_STAGES.index(stage) for stage in hyp]).astype(int)

    labels = np.zeros(n_windows * 30).astype(int)

    for start, step, stage in zip(start_index, end_index, hyp):
        labels[start : start + step] = stage * np.ones(step)

    labels = labels.reshape(n_windows, -1)
    labels = mode(labels, axis=1)[0]

    return signal.astype(np.float32), labels.astype(np.int16)


class DCSMPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[4, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="dcsm",
            signal_shape=[4, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        download_dir = self.dataset_folder
        url = "https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/dcsm_dataset.zip"

        if not os.path.exists(os.path.join(download_dir, "data", "sleep", "DCSM")):

            os.makedirs(
                os.path.join(download_dir, "data", "sleep", "DCSM"), exist_ok=True
            )

            zip_file = os.path.join(self.dataset_folder, "dcsm_dataset.zip")

            if not os.path.exists(zip_file):
                download_file(
                    url,
                    zip_file,
                )

            extract_large_zip(zip_file, download_dir)

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        return 578939

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        subjects_dir = os.path.join(self.dataset_folder, "data", "sleep", "DCSM")

        records = list(os.listdir(subjects_dir))
        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        return read_edf(
            os.path.join(self.dataset_folder, "data", "sleep", "DCSM", record)
        )

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


if __name__ == "__main__":

    p = DCSMPreprocessor(data_folder="/home/guido/physioex-data/")

    p.run()
