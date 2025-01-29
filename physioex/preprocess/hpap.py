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


class HPAPPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="hpap",
            signal_shape=[3, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[3, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records_dir = os.path.join(self.dataset_folder, "homepap", "polysomnography")

        edf_dir = os.path.join(records_dir, "edfs", "lab")

        # get the abs path of all the files into the edf_dir subdirectories
        records = []
        for root, dirs, files in os.walk(edf_dir):
            for f in files:
                record = os.path.join(root, f)

                basename = os.path.basename(f)
                basename = os.path.splitext(basename)[0]

                lab = "full" if "full" in record else "split"

                xml_path = os.path.join(
                    records_dir,
                    "annotations-events-nsrr",
                    "lab",
                    lab,
                    f"{basename}-nsrr.xml",
                )

                records.append((record, xml_path))

        return records

    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        edf_path, xml_path = record

        return process_sleepdata_file(edf_path, xml_path)

    def customize_table(self, table) -> pd.DataFrame:

        return table


if __name__ == "__main__":

    p = HPAPPreprocessor(data_folder="/mnt/vde/sleep-data/")

    p.run()
