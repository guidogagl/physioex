import os
import sys
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
import requests
from loguru import logger
from scipy.signal import resample
from tqdm import tqdm

from physioex.data.preprocessor import (Preprocessor, bandpass_filter,
                                        xsleepnet_preprocessing)


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


fs = 256

EVENTS = ["SLEEP-S0", "SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-REM"]
SLEEP_MAP = {"W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "REM": 4, "R": 4}


def read_edf(file_path):

    edf_path = file_path + ".edf"
    ann_path = file_path + ".txt"

    with open(ann_path, "r") as f:
        annotations = f.readlines()

    header = "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
    header_indx = annotations.index(header)
    # Remove the header
    annotations = annotations[header_indx + 1 :]

    # Split each line on the tab character
    annotations = [line.split("\t") for line in annotations]

    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(
        annotations,
        columns=[
            "Sleep Stage",
            "Position",
            "Time [hh:mm:ss]",
            "Event",
            "Duration[s]",
            "Location",
        ],
    ).dropna()
    # remove the rows corresponding to events not in EVENTS
    df = df[df["Event"].isin(EVENTS)]
    # map the sleep stages to the corresponding integer
    df["Sleep Stage"] = df["Sleep Stage"].map(SLEEP_MAP)
    labels = df["Sleep Stage"].values.astype(int)

    num_windows = labels.shape[0]

    try:
        f = pyedflib.EdfReader(edf_path)
    except:
        f.close()
        logger.warning(f"Error reading file {edf_path}, subject discarded")
        # print the error
        return None, None

    buffer = []

    for indx, modality in enumerate(["Fp2-F4", "ROC-LOC", "EMG1-EMG2", "ECG1-ECG2"]):

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


class CAPPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="cap",
            signal_shape=[4, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[4, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        download_dir = os.path.join(self.dataset_folder, "download")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

            zip_file = os.path.join(self.dataset_folder, "cap_dataset.zip")

            if not os.path.exists(zip_file):
                download_file(
                    "https://physionet.org/static/published-projects/capslpdb/cap-sleep-database-1.0.0.zip",
                    zip_file,
                )

            extract_large_zip(zip_file, download_dir)

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        return 137243

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        subjects_dir = os.path.join(
            self.dataset_folder, "download", "cap-sleep-database-1.0.0"
        )

        records_file = os.path.join(subjects_dir, "gender-age.xlsx")

        records = pd.read_excel(records_file, header=None)[0].values

        records = [record.lower() for record in records]

        for i in range(len(records)):
            if "sbd" in records[i]:
                records[i] = records[i].replace("sbd", "sdb")

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        return read_edf(
            os.path.join(
                self.dataset_folder, "download", "cap-sleep-database-1.0.0", record
            )
        )

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        my_table = os.path.join(
            self.dataset_folder,
            "download",
            "cap-sleep-database-1.0.0",
            "gender-age.xlsx",
        )
        my_table = pd.read_excel(my_table, header=None)

        table["age"] = my_table[2].values
        table["sex"] = my_table[1].values

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

    p = CAPPreprocessor(data_folder="/home/guido/shared/")

    p.run()
