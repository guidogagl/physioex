import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
import rarfile
import requests
from loguru import logger
from pyunpack import Archive
from scipy.signal import resample
from tqdm import tqdm

from physioex.data.preprocessor import Preprocessor, xsleepnet_preprocessing

fs = 200

subgroups = [
    {"id": "subgroupI", "subjects": list(range(1, 101))},
    {"id": "subgroupIII", "subjects": list(range(1, 11))},
]

subgroups[0]["subjects"].remove(40)


class ISRUCPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="isruc",
            signal_shape=[1, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[1, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None:
        download_dir = os.path.join(self.dataset_folder, "download")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

        for subgroup in subgroups:

            subgroup_path = os.path.join(download_dir, subgroup["id"])

            if not os.path.exists(subgroup_path):
                os.makedirs(subgroup_path, exist_ok=True)

            for subject in tqdm(subgroup["subjects"]):
                subject_path = os.path.join(subgroup_path, str(subject))

                if not os.path.exists(subject_path):
                    os.makedirs(subject_path, exist_ok=True)

                    file_path = os.path.join(subgroup_path, "tmp.rar")

                    url = f"http://dataset.isr.uc.pt/ISRUC_Sleep/{subgroup['id']}/{subject}.rar"

                    download_file(url, file_path)

                    with rarfile.RarFile(file_path, "r") as rar_ref:
                        rar_ref.extractall(subgroup_path)

                    # Rimuovi il file rar
                    os.remove(file_path)

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        return 98201

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records = []

        for subgroup in subgroups:
            for subject in subgroup["subjects"]:
                records.append(
                    os.path.join(
                        self.dataset_folder,
                        "download",
                        subgroup["id"],
                        str(subject),
                        str(subject),
                    )
                )

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        return read_edf(record)

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        sub1_desc = pd.read_excel(
            "http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_I_Submission.xlsx",
            header=2,
        )
        sub3_desc = pd.read_excel(
            "http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_III_Submission.xlsx",
            header=2,
        )

        # todo: remove discarded subjects from sub1desc
        sub1_desc = sub1_desc[sub1_desc["Subject"] != 40]
        sub1_desc["Subject"] = list(range(sub1_desc.shape[0]))

        sub3_desc["Subject"] = sub3_desc["Subject"] - 1 + sub1_desc.shape[0]

        frames = [sub1_desc, sub3_desc]
        desc = pd.concat(frames).reset_index(drop=True)

        table["age"] = desc["Age"]
        table["gender"] = desc["Sex"]

        return table

    @logger.catch
    def get_sets(self) -> Tuple[np.array, np.array, np.array]:

        np.random.seed(42)

        table = self.table.copy()

        np.random.seed(42)

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


def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def get_labels(filepath):
    # open the txt file to get the labels
    with open(filepath + "_1.txt", "r") as f:
        labels = f.readlines()

    labels = np.array(
        [int(label.strip()) for label in labels if label.strip().isdigit()]
    ).astype(np.int16)
    return labels


mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


@logger.catch
def read_edf(file_path):
    try:
        labels = pd.read_excel(file_path + "_1.xlsx")["Stage"].dropna()
    except:
        labels = pd.read_excel(file_path + "_1.xlsx", header=None)[1].dropna()

    # Stampa gli elementi unici di labels che non sono in mapping
    not_in_mapping = labels[~labels.isin(mapping.keys())].unique()

    # map the labels to the new values using pandas
    labels = labels[labels.isin(mapping.keys())]
    labels = labels.map(mapping).values

    num_windows = len(labels)

    f = pyedflib.EdfReader(file_path + ".rec")

    try:
        i = f.getSignalLabels().index("C3-A2")
    except:
        try:
            i = f.getSignalLabels().index("C3-M2")
        except:
            logger.warning(
                f"No valid channels found for {file_path} available are {f.getSignalLabels()}"
            )
            exit()
            return None, None

    signal = f.readSignal(i)

    f._close()

    # windowing of the signal
    signal = signal.reshape(-1, fs)

    if num_windows != signal.shape[0] // 30:
        logger.warning("Number of windows mismatch")
        print("Label estimated windows ", num_windows)
        print("Signal esitmated windows ", signal.shape[0] // 30)
        num_windows = min(num_windows, signal.shape[0] // 30)
        labels = labels[:num_windows]
        print(f"Elementi unici in labels che non sono in mapping: {not_in_mapping}")

    signal = signal[: num_windows * 30]
    signal = signal.reshape(num_windows, 30 * fs)

    signal = resample(signal, num=3000, axis=1)

    buffer = signal.reshape(num_windows, 1, 3000)

    return buffer, labels


if __name__ == "__main__":

    p = ISRUCPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
