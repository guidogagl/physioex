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

from physioex.data.preprocessor import (
    Preprocessor,
    bandpass_filter,
    xsleepnet_preprocessing,
)
from physioex.data.svuh.constant import (
    SVUH_NUM_WINDOWS,
    signal_shape,
    url,
    xsleepnet_shape,
)


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


def read_edf(file_path):

    # open the txt file to get the labels
    with open(file_path + "_stage.txt", "r") as f:
        labels = f.readlines()

    # remove \n from the labels
    labels = list(map(lambda x: int(x.strip()), labels))
    labels = np.array(labels)

    n_samples = labels.shape[0]

    f = pyedflib.EdfReader(file_path + ".rec")

    buffer = []

    for indx, modality in enumerate(["C3A2", "EOG", "EMG", "ECG"]):
        if modality == "EOG":

            left = f.getSignalLabels().index("Lefteye")
            right = f.getSignalLabels().index("RightEye")

            if f.getSampleFrequency(left) != f.getSampleFrequency(right):
                logger.error("Sampling frequency of EOG signals is different")
                exit()

            fs = int(f.getSampleFrequency(left))

            signal = (f.readSignal(left) - f.readSignal(right)).reshape(-1, fs)
            num_windows = signal.shape[0] // 30

            # print( f"Modality {modality} fs {fs} num windows {num_windows} shape {signal.shape}" )

            signal = signal[: num_windows * 30]
            signal = signal.reshape(-1, 30 * fs)

        else:
            i = f.getSignalLabels().index(modality)
            fs = int(f.getSampleFrequency(i))
            signal = f.readSignal(i).reshape(-1, fs)

            # print( f"Modality {modality} fs {fs} shape {signal.shape}" )
            # consider windows of 30 seconds, discard the last epoch if not fit
            num_windows = signal.shape[0] // 30

            # print( f"Modality {modality} fs {fs} num windows {num_windows} shape {signal.shape}" )

            signal = signal[: num_windows * 30]
            signal = signal.reshape(-1, 30 * fs)

        # resample the signal at 100Hz
        signal = resample(signal, num=30 * 100, axis=1)
        # pass band the signal between 0.3 and 40 Hz
        signal = bandpass_filter(signal, 0.3, 40, 100)

        buffer.append(signal)
    f._close()

    buffer = np.array(buffer)
    n_samples = min(n_samples, buffer.shape[1])

    buffer, labels = buffer[:, :n_samples, :], labels[:n_samples]

    mask = np.logical_and(labels != 6, labels != 7)
    buffer, labels = buffer[:, mask], labels[mask]

    # map the labels to the new values
    labels = np.array(
        list(
            map(
                lambda x: (
                    0
                    if x == 0
                    else 4 if x == 1 else 1 if x == 2 else 2 if x == 3 else 3
                ),
                labels,
            )
        )
    )

    # print( f"Buffer shape {buffer.shape} labels shape {labels.shape}" )
    buffer = np.transpose(buffer, (1, 0, 2))
    return buffer, labels


class SVUHPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="svuh",
            signal_shape=signal_shape,
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[xsleepnet_shape],
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        download_dir = os.path.join(self.dataset_folder, "download")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

            zip_file = os.path.join(self.dataset_folder, "svuh_dataset.zip")

            if not os.path.exists(zip_file):
                download_file(
                    url,
                    zip_file,
                )

            extract_large_zip(zip_file, download_dir)

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        return SVUH_NUM_WINDOWS

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        subjects_dir = os.path.join(self.dataset_folder, "download", "files")

        records = pd.read_excel(os.path.join(subjects_dir, "SubjectDetails.xls"))[
            "Study Number"
        ].values

        records = [record.lower() for record in records]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        return read_edf(os.path.join(self.dataset_folder, "download", "files", record))

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        subjects_dir = os.path.join(self.dataset_folder, "download", "files")
        desc = pd.read_excel(os.path.join(subjects_dir, "SubjectDetails.xls"))

        table["sex"] = desc["Gender"]
        table["age"] = desc["Age"]

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

    p = SVUHPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
