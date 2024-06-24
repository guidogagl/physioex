import os
import shutil
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from psg_utils.io.header import extract_header
from psg_utils.io.high_level_file_loaders import load_psg
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm
from wfdb.io import rdann

from physioex.data.constant import get_data_folder
from physioex.data.preprocessor import (Preprocessor, bandpass_filter,
                                        xsleepnet_preprocessing)


class PHYSIOPreprocessor(Preprocessor):

    def __init__(self, data_folder: str = None):

        super().__init__(
            dataset_name="physio2018",
            signal_shape=[3, 3000],
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[[3, 29, 129]],
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records_dir = os.path.join(self.dataset_folder, "download", "training")

        filenames = os.listdir(records_dir)

        # take only the directories
        filenames = [f for f in filenames if os.path.isdir(os.path.join( records_dir , f))]

        return filenames

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        LABEL_MAP = {
            'N1': 1,
            'N2': 2,
            'N3': 3,
            'R': 4,
            'W': 0,
        }

        record_dir = os.path.join(self.dataset_folder, "download", "training", record)
        record_files = os.listdir(record_dir)
        filepath = [ record for record in record_files if record.endswith(".mat") ][0]
        hyp_file = [ record for record in record_files if record.endswith(".arousal") ][0]
        
        filepath = os.path.join(record_dir, filepath)                        
        signal, header = load_psg(filepath, load_channels=['C3-M2', "E1-M2", "Chin1-Chin2", "ECG"])
        
        sample_rate = header['sample_rate']
        
        filepath = os.path.join(record_dir, hyp_file)                
        hyp_file = filepath + ".st"
                
        if not os.path.exists( hyp_file ):
            shutil.copyfile( filepath, hyp_file)
                
        hyp = rdann( filepath, "st" )
        
        pairs = zip(hyp.aux_note, hyp.sample)
        stages = [s for s in pairs if not ("(" in s[0] or ")" in s[0])]
        stages = [(s[0], int(s[1]/sample_rate)) for s in stages]
        stages, starts = map(list, zip(*stages))
        stages = [LABEL_MAP[s] for s in stages]
        
        if starts[0] != 0:
            stages = [ -1 ] + stages
                        
        num_windows = len( stages )
        signal = signal[ : num_windows * 30 * sample_rate].astype(np.float32)
        signal = np.reshape( signal, ( num_windows, 30 * sample_rate, 4 ) )
        stages = np.array( stages )
        
        if stages[0] == -1:
            stages = stages[1:]
            signal = signal[1:]
                
        signal = resample(signal, num=3000, axis = 1)
        signal = np.transpose( signal, (0, 2, 1))
        
        signal = bandpass_filter(signal, 0.3, 40, 100)
        
        return signal, stages.astype(int)

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
        num_windows = super().get_dataset_num_windows()
        print(num_windows)
        exit()
        return 


if __name__ == "__main__":

    p = PHYSIOPreprocessor(data_folder="/home/guido/shared/")

    p.run()
