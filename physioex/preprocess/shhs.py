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

from physioex.preprocess.utils.sleepdata import process_sleepdata_file


class SHHSPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="shhs",
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

        table = os.path.join(
            self.dataset_folder,
            "shhs_raw",
            "datasets",
            "shhs-harmonized-dataset-0.21.0.csv",
        )
        table = pd.read_csv(table)

        # take visit 1 only

        table = table[table["visitnumber"] == 1]

        nsrrids = list(table["nsrrid"].values.astype(int))

        edf_path = os.path.join(
            self.dataset_folder, "shhs_raw", "polysomnography", "edfs", "shhs1"
        )
        ann_path = os.path.join(
            self.dataset_folder,
            "shhs_raw",
            "polysomnography",
            "annotations-events-nsrr",
            "shhs1",
        )

        get_edf_path = lambda nsrrid: os.path.join(edf_path, f"shhs1-{nsrrid}.edf")
        get_ann_path = lambda nsrrid: os.path.join(ann_path, f"shhs1-{nsrrid}-nsrr.xml")

        records = [
            (nsrrid, get_edf_path(nsrrid), get_ann_path(nsrrid)) for nsrrid in nsrrids
        ]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        nsrrid, edf_path, ann_path = record

        signal, labels = process_sleepdata_file(edf_path, ann_path)

        return signal, labels

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        table_ = os.path.join(
            self.dataset_folder,
            "shhs_raw",
            "datasets",
            "shhs-harmonized-dataset-0.21.0.csv",
        )
        table_ = pd.read_csv(table_)

        # take visit 1 only
        table_ = table_[table_["visitnumber"] == 1]
        table_["nsrrid"] = table_["nsrrid"].astype(int)

        nsrrids = list(table_["nsrrid"])
        subject_id = [i for i, _ in enumerate(nsrrids)]

        table_["subject_id"] = subject_id

        table_ = table_.set_index("subject_id")

        # join table and table_ on subject_id column
        table = table.join(table_, on="subject_id")

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


if __name__ == "__main__":

    p = SHHSPreprocessor(data_folder="/mnt/guido-data/")

    p.run()
