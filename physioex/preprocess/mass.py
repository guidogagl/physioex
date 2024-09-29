import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing


class MASSPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="mass",
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
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
            if file.endswith(".mat") and file.startswith("SS0"):
                subject_id = file[:-8]
                if subject_id not in subjects:
                    subjects.append(subject_id)

        return subjects

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        signal = []
        for modality in ["eeg", "eog", "emg"]:
            filepath = os.path.join(
                self.dataset_folder, "mat", f"{record}_{modality}.mat"
            )

            with h5py.File(filepath, "r") as f:
                data = {key: f[key][()] for key in f.keys()}

            signal.append(np.transpose(data["X1"], (1, 0)).astype(np.float32))

            if modality == "eeg":
                labels = np.reshape(data["label"], (-1)).astype(np.int16)

        signal = np.array(signal)
        signal = np.transpose(signal, (1, 0, 2))

        labels = labels - 1

        if min(labels) < 0 or max(labels) > 4:
            logger.error(f"Invalid labels for record {record}")
            logger.error(f"Min label: {min(labels)}")
            logger.error(f"Max label: {max(labels)}")
            exit()

        return signal, labels

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        # this method should be provided by the user
        # the method should return a customized version of the dataset table before saving it
        return table

    @logger.catch
    def get_sets(self) -> Tuple[List, List, List]:
        url = (
            "https://github.com/pquochuy/xsleepnet/raw/master/mass/data_split_eval.mat"
        )
        matpath = os.path.join(self.dataset_folder, "tmp.mat")
        urlretrieve(url, matpath)

        split_matrix = loadmat(matpath)
        subjects_records = self.get_subjects_records()
        subjects = list(range(len(subjects_records)))

        test_subjects, valid_subjects, train_subjects = [], [], []

        for fold in range(len(split_matrix["test_sub"])):
            te_subjects = np.array(list(split_matrix["test_sub"][fold][0][0])).astype(
                np.int16
            )
            v_subjects = np.array(list(split_matrix["eval_sub"][fold][0][0])).astype(
                np.int16
            )
            tr_subjects = list(set(subjects) - set(v_subjects) - set(te_subjects))
            tr_subjects = np.array(tr_subjects).astype(np.int16)

            test_subjects.append(te_subjects)
            valid_subjects.append(v_subjects)
            train_subjects.append(tr_subjects)

        os.remove(matpath)

        return train_subjects, valid_subjects, test_subjects


if __name__ == "__main__":

    p = MASSPreprocessor(data_folder="/mnt/guido-data/")

    p.run()
