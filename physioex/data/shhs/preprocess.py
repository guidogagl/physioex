import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.io import loadmat
from tqdm import tqdm

from physioex.data.constant import get_data_folder
from physioex.data.preprocessor import Preprocessor, xsleepnet_preprocessing


class SHHSPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="shhs",
            signal_shape=[3, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[3, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # this method should be provided by the user
        # the method should return a list containing the path of each subject record
        # each path is needed to be passed as argument to the function read_subject_record(self, record)

        mat_dir = os.path.join(self.dataset_folder, "mat")

        filenames = os.listdir(mat_dir)

        subjects = []
        for file in filenames:
            if file.endswith(".mat") and file.startswith("n"):
                subject_id = file.split("_")[0][1:]
                if subject_id not in subjects:
                    subjects.append(subject_id)

        return subjects

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        signal = []
        for modality in ["eeg", "eog", "emg"]:
            filepath = os.path.join(
                self.dataset_folder, "mat", f"n{record}_{modality}.mat"
            )

            with h5py.File(filepath, "r") as f:
                data = {key: f[key][()] for key in f.keys()}

            signal.append(np.transpose(data["X1"], (1, 0)).astype(np.float32))

            if modality == "eeg":
                labels = np.reshape(data["label"], (-1)).astype(np.int16)

        signal = np.array(signal)
        signal = np.transpose(signal, (1, 0, 2))

        return signal, labels

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        # this method should be provided by the user
        # the method should return a customized version of the dataset table before saving it
        return table

    @logger.catch
    def get_sets(self) -> Tuple[List, List, List]:
        url = "https://github.com/pquochuy/SleepTransformer/raw/main/shhs/data_split_eval.mat"
        matpath = os.path.join(self.dataset_folder, "tmp.mat")
        urlretrieve(url, matpath)

        split_matrix = loadmat(matpath)

        test_subjects = np.array(split_matrix["test_sub"][0].astype(np.int16))
        valid_subjects = np.array(split_matrix["eval_sub"][0].astype(np.int16))

        subjects_records = self.get_subjects_records()
        subjects = list(range(len(subjects_records)))

        train_subjects = list(set(subjects) - set(valid_subjects) - set(test_subjects))
        train_subjects = np.array(train_subjects).astype(np.int16)

        os.remove(matpath)

        return (
            train_subjects.reshape(1, -1),
            valid_subjects.reshape(1, -1),
            test_subjects.reshape(1, -1),
        )

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        # this method should be provided by the user
        # the method should return a int number representing the total number of windows in the dataset. For efficiency the best would be to precompute this value.
        return 5457206


if __name__ == "__main__":

    p = SHHSPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
