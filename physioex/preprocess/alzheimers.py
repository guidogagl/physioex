import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve
import shutil 

import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, resample
import pyedflib
from sklearn.model_selection import StratifiedKFold, train_test_split

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing

from physioex.preprocess.utils.sleepdata import get_channel_from_available, get_channels, read_channel_signal

POSSIBLE_EEG_CHANNELS = [
    # "EEG C4-A1",
    "EEG C4-REF",
    # "EEG C3-A2", 
    "EEG C3-REF" 
]

POSSIBLE_EOG_CHANNELS = [
    ('EEG EOG1-REF', 'EEG EOG2-REF')
]

POSSIBLE_EMG_CHANNELS = [
    "EMG CHIN",
    "EMG1",
    "EMG2",
    "EMG3",
    "EMG4",
    "EMG5"
]

def read_edf(edf_path, tsv_path):

    stages_map = {
        "Wake": 0,
        "S1": 1,
        "S2": 2, 
        "S3": 3, 
        "REM": 4
    }

    df = pd.read_csv(tsv_path, sep='\t', usecols=(0,1,2), names=('start', 'end', 'stage'), skiprows=8)
    stages = [stages_map[s] for s in df['stage']]
    scores_start = df.iloc[0]['start']
    scores_end = df.iloc[-1]['end']
    
    fs=100
    epoch_second=30

    available_channels = get_channels(edf_path)
    eeg_channel = get_channel_from_available(available_channels, POSSIBLE_EEG_CHANNELS)
    if eeg_channel is None:
        print(f"Error: no EEG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eeg, old_fs = read_channel_signal(edf_path, eeg_channel, scores_start, scores_end - scores_start)

    # Creazione del filtro FIR bandpass
    Nfir = 500
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

    # Applicazione del filtro al segnale EEG
    eeg = filtfilt(b_band, 1, eeg)

    if fs != old_fs:
        eeg = resample(eeg, int(len(eeg) * fs / old_fs))

    eog_channel = get_channel_from_available(available_channels, POSSIBLE_EOG_CHANNELS)
    if eog_channel is None:
        print(f"Error: no EOG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eog, old_fs = read_channel_signal(edf_path, eog_channel, scores_start, scores_end - scores_start)

    # filtering and resampling
    eog = filtfilt(b_band, 1, eog)

    if fs != old_fs:
        eog = resample(eog, int(len(eog) * fs / old_fs))

    emg_channel = get_channel_from_available(available_channels, POSSIBLE_EMG_CHANNELS)
    if emg_channel is None:
        print(f"Error: no EMG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        emg, old_fs = read_channel_signal(edf_path, emg_channel, scores_start, scores_end - scores_start)

    # filtering and resampling
    b_band = firwin(Nfir + 1, 10, pass_zero=False, fs=old_fs)
    emg = filtfilt(b_band, 1, emg)

    if fs != old_fs:
        emg = resample(emg, int(len(emg) * fs / old_fs))

    # buffer the signals into epochs
    signal = np.array([eeg, eog, emg])
    signal = np.transpose(signal).reshape(len(stages), epoch_second * fs, 3)

    # find the epochs associated with stages < 0 or > 5
    stages = np.array(stages)

    signal = np.transpose(signal, (0, 2, 1))

    return signal.astype(np.float32), stages.astype(int)    


class AlzheimersPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="alzheimers",
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

        self.raw_data_folder = os.path.join(self.dataset_folder, 'Data')
        self.original_subject = []
        self.group = []

        remove_subject = os.path.join(self.raw_data_folder, 'AD014') # edf file is corrupt
        if os.path.exists(remove_subject):
            shutil.rmtree(remove_subject)  


    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records_dir = os.path.join(self.raw_data_folder)
        records = os.listdir(records_dir)

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        for file in os.listdir(os.path.join(self.raw_data_folder, record)):
            if file.endswith(".edf"):
                edf_path = os.path.join(self.raw_data_folder, record, file)
            elif file.endswith(".tsv"):
                tsv_path = os.path.join(self.raw_data_folder, record, file)
        
        signal, labels = read_edf(edf_path, tsv_path)

        if signal is not None and labels is not None:
            self.original_subject.append(record)
            self.group.append(record[:2])

        return signal, labels

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        """
        Customizes the dataset table before saving it.

        (Optional) Method to be provided by the user.

        Parameters:
            table (pd.DataFrame): The dataset table to be customized.

        Returns:
            pd.DataFrame: The customized dataset table.
        """

        table['original_subject'] = self.original_subject
        table['group'] = self.group

        return table

    @logger.catch
    def get_sets(self, n_folds=4) -> Tuple[np.array, np.array, np.array]:
        """
        Returns the train, validation, and test subjects. Stratified per patient group.

        Returns:
            Tuple[np.array, np.array, np.array]: A tuple containing the train, validation, and test subjects.
        """

        random_state = 42

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        X = self.table['subject_id']
        y = self.table['group']

        train_subjects = []
        valid_subjects = []
        test_subjects = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
            # Split indices
            df_train_val = self.table.iloc[train_val_idx]
            df_test = self.table.iloc[test_idx]

            # Now split train_val into train and val (again stratified)
            df_train, df_val = train_test_split(
                df_train_val,
                test_size=0.1765,  # 0.15 / (0.15 + 0.7)
                stratify=df_train_val['group'],
                random_state=random_state
            )

            train_subjects.append(df_train['subject_id'].to_numpy())
            valid_subjects.append(df_val['subject_id'].to_numpy())
            test_subjects.append(df_test['subject_id'].to_numpy())

        return train_subjects, valid_subjects, test_subjects



if __name__ == "__main__":

    p = AlzheimersPreprocessor(data_folder="/home/coder/sleep/sleep-data/")

    p.run()
