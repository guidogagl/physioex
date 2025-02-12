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


class MESAPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="mesa",
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records_dir = os.path.join(self.dataset_folder, "raw_mesa", "polysomnography")

        edf_dir = os.path.join(records_dir, "edfs")

        # get the abs path of all the files into the edf_dir subdirectories
        records = []
        for root, dirs, files in os.walk(edf_dir):
            for f in files:
                record = os.path.join(root, f)

                basename = os.path.basename(f)
                basename = os.path.splitext(basename)[0]

                xml_path = os.path.join(
                    records_dir, "annotations-events-nsrr", f"{basename}-nsrr.xml"
                )

                records.append((record, xml_path))

        return records

    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        edf_path, xml_path = record

        return process_sleepdata_file(edf_path, xml_path)

    def customize_table(self, table) -> pd.DataFrame:

        return table


if __name__ == "__main__":

    p = MESAPreprocessor(data_folder="/mnt/vde/sleep-data/")

    p.run()
