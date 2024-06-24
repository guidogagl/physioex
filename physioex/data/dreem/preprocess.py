import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from dirhash import dirhash
from loguru import logger
from tqdm import tqdm

import physioex.data.dreem.utils as utl
from physioex.data.constant import get_data_folder, set_data_folder
from physioex.data.preprocessor import Preprocessor, xsleepnet_preprocessing

picks = ["C3_M2", "EOG", "EMG"]


class DREEMPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        self.download_dir = os.path.join(
            get_data_folder() if data_folder is None else set_data_folder(data_folder),
            "dreem",
            "h5",
        )

        logger.info("Pre-fetching the dataset")
        try:
            found = (
                str(dirhash(self.download_dir, "md5", jobs=os.cpu_count()))
                == utl.DATASET_HASH
            )
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            found = False

        if not found:
            os.makedirs(self.download_dir, exist_ok=True)
            logger.info("Data not found, download dataset...")
            utl.download_dreem_dataset(self.download_dir)

        self.version = None

        super().__init__(
            dataset_name="dreem",
            signal_shape=[3, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[3, 29, 129]],
            data_folder=data_folder,
        )

    def run(self):
        dataset_folder = self.dataset_folder

        self.version = "dodh"
        self.dataset_folder = os.path.join(dataset_folder, self.version)

        os.makedirs(self.dataset_folder, exist_ok=True)

        logger.info(f"Processing dataset version {self.version}")
        super().run()

        self.version = "dodo"
        self.dataset_folder = os.path.join(dataset_folder, self.version)
        os.makedirs(self.dataset_folder, exist_ok=True)

        logger.info(f"Processing dataset version {self.version}")
        super().run()

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        dodh = 24662
        dodo = 54197
        return dodh if self.version == "dodh" else dodo

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        download_dir = os.path.join(self.download_dir, self.version)

        files = [f for f in os.listdir(download_dir) if f.endswith(".h5")]

        return files

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        subj = h5py.File(os.path.join(self.download_dir, self.version, record), "r")
        hyp = np.array(subj["hypnogram"][:], dtype=int)

        n_win = len(hyp)

        X, y = [], []

        for i in range(n_win):

            if hyp[i] == -1:
                continue

            C3_M2 = subj["signals"]["eeg"]["C3_M2"][
                i * (250 * 30) : (i + 1) * (250 * 30)
            ]

            EOG = subj["signals"]["eog"]["EOG1"][i * (250 * 30) : (i + 1) * (250 * 30)]

            EMG = subj["signals"]["emg"]["EMG"][i * (250 * 30) : (i + 1) * (250 * 30)]

            for preprocess in utl.preprocessors:
                C3_M2 = preprocess(C3_M2)
                EOG = preprocess(EOG)
                EMG = preprocess(EMG)

            X.append(np.array([C3_M2, EOG, EMG]))
            y.append(hyp[i])

        X = np.array(X)
        y = np.array(y).reshape(-1)

        return X, y

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
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


if __name__ == "__main__":

    p = DREEMPreprocessor(data_folder="/esat/biomeddata/ggagliar/")
    p.run()
