import os
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
import requests
from loguru import logger
from scipy.signal import resample
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


stages_map = [
    "Sleep stage W",
    "Sleep stage N1",
    "Sleep stage N2",
    "Sleep stage N3",
    "Sleep stage R",
]

fs = 256


def read_edf(file_path):

    stages_path = file_path[:-4] + "_sleepscoring.edf"
    f = pyedflib.EdfReader(stages_path)
    _, _, annt = f.readAnnotations()
    f._close()

    annotations = []
    for a in annt:
        if a in stages_map:
            annotations.append(stages_map.index(a))

    labels = np.reshape(np.array(annotations).astype(int), (-1))

    num_windows = labels.shape[0]

    f = pyedflib.EdfReader(file_path)
    buffer = []

    for indx, modality in enumerate(["EEG C3-M2", "EOG", "EMG chin", "ECG"]):
        if modality == "EOG":

            left = f.getSignalLabels().index("EOG E1-M2")
            right = f.getSignalLabels().index("EOG E2-M2")

            signal = (f.readSignal(left) + f.readSignal(right)).reshape(-1, fs)

        else:

            i = f.getSignalLabels().index(modality)
            fs = int(f.getSampleFrequency(i))
            signal = f.readSignal(i).reshape(-1, fs)

        signal = signal[: num_windows * 30]
        signal = signal.reshape(-1, 30 * fs)

        # resample the signal at 100Hz
        signal = resample(signal, num=30 * 100, axis=1)
        # pass band the signal between 0.3 and 40 Hz
        signal = bandpass_filter(signal, 0.3, 40, 100)

        buffer.append(signal)
    f._close()

    buffer = np.array(buffer)
    buffer = np.transpose(buffer, (1, 0, 2))

    return buffer, labels


class HMCPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[4, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="hmc",
            signal_shape=[4, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        download_dir = os.path.join(self.dataset_folder, "download")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

            zip_file = os.path.join(self.dataset_folder, "hmc_dataset.zip")

            if not os.path.exists(zip_file):
                download_file(
                    "https://physionet.org/static/published-projects/hmc-sleep-staging/haaglanden-medisch-centrum-sleep-staging-database-1.1.zip",
                    zip_file,
                )

            extract_large_zip(zip_file, download_dir)

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        subjects_dir = os.path.join(
            self.dataset_folder,
            "download",
            "haaglanden-medisch-centrum-sleep-staging-database-1.1",
        )

        records_file = os.path.join(subjects_dir, "RECORDS")

        with open(records_file, "r") as file:
            records = file.readlines()

        records = [record.rstrip("\n") for record in records]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        return read_edf(
            os.path.join(
                self.dataset_folder,
                "download",
                "haaglanden-medisch-centrum-sleep-staging-database-1.1",
                record,
            )
        )

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
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


if __name__ == "__main__":

    p = HMCPreprocessor(data_folder="/mnt/guido-data/")

    p.run()
