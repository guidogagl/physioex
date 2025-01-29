import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
from loguru import logger
from scipy.signal import resample

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing
from physioex.preprocess.utils.sleepdata import process_sleepdata_file


class MROSPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="mros",
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        table = os.path.join(
            self.dataset_folder,
            "mros_raw",
            "datasets",
            "mros-visit1-harmonized-0.6.0.csv",
        )
        table = pd.read_csv(table)

        # take visit 1 only

        table = table[table["visit"] == 1]

        nsrrids = list(table["nsrrid"].values.astype(str))
        # convert to lowercase
        nsrrids = [nsrrid.lower() for nsrrid in nsrrids]

        edf_path = os.path.join(
            self.dataset_folder, "mros_raw", "polysomnography", "edfs", "visit1"
        )
        ann_path = os.path.join(
            self.dataset_folder,
            "mros_raw",
            "polysomnography",
            "annotations-events-nsrr",
            "visit1",
        )

        get_edf_path = lambda nsrrid: os.path.join(
            edf_path, f"mros-visit1-{nsrrid}.edf"
        )
        get_ann_path = lambda nsrrid: os.path.join(
            ann_path, f"mros-visit1-{nsrrid}-nsrr.xml"
        )

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
            "mros_raw",
            "datasets",
            "mros-visit1-harmonized-0.6.0.csv",
        )
        table_ = pd.read_csv(table_)

        # take visit 1 only
        table_ = table_[table_["visit"] == 1]
        nsrrids = table_["nsrrid"].values.astype(str)
        nsrrids = [nsrrid.lower() for nsrrid in nsrrids]
        table_["nsrrid"] = nsrrids

        subject_id = [i for i, _ in enumerate(nsrrids)]

        table_["subject_id"] = subject_id

        table_ = table_.set_index("subject_id")

        # join table and table_ on subject_id column
        table = table.join(table_, on="subject_id")

        return table


if __name__ == "__main__":

    p = MROSPreprocessor(data_folder="/mnt/vde/sleep-data/")

    p.run()
