import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

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
    "EEG C4-A1",
    "EEG C4-REF",
    "EEG C3-A2", 
    "EEG C3-REF" 
]

POSSIBLE_EOG_CHANNELS = [
    ("EOG LEFT", "EOG RIGHT"),
    ("EOG2", "EOG1")
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

    df = pd.read_csv(tsv_path, sep='\t', header=None)
    
    lights_idx = df.index[(df[2] == 'LIGHTS_OFF') | (df[2] == 'LIGHT_OFF')]
    if len(lights_idx)>0:   
        df.drop(np.arange(lights_idx[0]+1), inplace=True)
    df = df.iloc[:-1] # last epoch is never a whole epoch for some reason
    stages = df[2].tolist()
    stages = [stages_map[s] if s in stages_map.keys() else -1 for s in stages]

    scores_start = int(df[0].iloc[0])
    scores_end = int(df[1].iloc[-1])
    
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

    # find the epochs associated with stages < 0 or >= 5
    stages = np.array(stages)
    invalid_epochs = np.where(np.logical_or(stages < 0, stages >= 5))[0]

    # remove the invalid epochs
    stages = np.delete(stages, invalid_epochs)
    signal = np.delete(signal, invalid_epochs, axis=0)

    # do not consider recording if only wake data
    if all(s == 0 for s in stages):
        return None, None

    # remove Wake epochs if Wake is the biggest class:
    count_stage = np.bincount(stages)
    if count_stage[0] > max(count_stage[1:]):  # if too much W
        # print('Wake is the biggest class. Trimming it..')
        second_largest = max(count_stage[1:])

        W_ind = stages == 0  # W indices
        last_evening_W_index = np.where(np.diff(W_ind) != 0)[0][0] + 1
        if stages[0] == 0:  # only true if the first epoch is W
            num_evening_W = last_evening_W_index
        else:
            num_evening_W = 0

        first_morning_W_index = np.where(np.diff(W_ind) != 0)[0][-1] + 1
        num_morning_W = len(stages) - first_morning_W_index + 1

        nb_pre_post_sleep_wake_eps = num_evening_W + num_morning_W
        if nb_pre_post_sleep_wake_eps > second_largest:
            total_W_to_remove = nb_pre_post_sleep_wake_eps - second_largest
            if num_evening_W > total_W_to_remove:
                stages = stages[total_W_to_remove:]
                signal = signal[total_W_to_remove:]
            else:
                evening_W_to_remove = num_evening_W
                morning_W_to_remove = total_W_to_remove - evening_W_to_remove
                stages = stages[evening_W_to_remove : len(stages) - morning_W_to_remove]
                signal = signal[evening_W_to_remove : len(signal) - morning_W_to_remove]

        # print(f'New stages distribution: {np.bincount(stages)}')
    else:
        # print('Wake is not the biggest class, nothing to remove.')
        pass

    signal = np.transpose(signal, (0, 2, 1))

    return signal.astype(np.float32), stages.astype(int)    


class ParkinsonsPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
        night : bool = True,
        healthy : bool = True
    ):

        night_str = "night" if night else "nap"
        healthy_str = "HOA" if healthy else "PD"
        
        dataset_name = f"parkinsons4/{night_str}/{healthy_str}"
        
        super().__init__(
            dataset_name=dataset_name,
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

        self.night = night
        self.healthy = healthy
        
        self.root_folder = os.path.join(self.data_folder, "parkinsons4" )
        
    @logger.catch
    def get_subjects_records(self) -> List[str]:
        
        table_path = os.path.join(self.root_folder, "Parkinson_data", "Target_sleep_demographic.csv")
        self.demographics_table = pd.read_csv( table_path )
        
        table = self.demographics_table.copy()[["record_id", "group"]]
        table = table.sort_values(by="record_id").reset_index(drop=True)

        # now loop into the couples "record_id" and "group" to create the records list
        records_list = []
        records_dict = {}
        count = 0
        for idx, row in table.iterrows():
            subject_id = row['record_id']
            group = row['group']
            
            if self.healthy and group != "HOA":
                continue
            if not self.healthy and group != "PD":
                continue
            
            records_dir = os.path.join(self.root_folder, "Parkinson_data", "Data", str(subject_id) + "_nap" if not self.night else str(subject_id) )
            # check if the directory exists if not continue
            if not os.path.exists(records_dir):
                logger.warning(f"Record directory {records_dir} does not exist. Skipping subject {subject_id}.")
                continue

            records_list.append(records_dir)
            records_dict[count] = subject_id
            count += 1

        self.records_dict = records_dict

        return records_list

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        
        files = os.listdir(record)
        edf_path = [f for f in files if f.endswith('.edf')][0]
        tsv_path = [f for f in files if f.endswith('.tsv')][0]
                
        signal, labels = read_edf(os.path.join(record, edf_path), os.path.join(record, tsv_path))
        return signal, labels
    
    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:

        table["record_id"] = table["subject_id"].map(self.records_dict)
        
        return table


def global_split(data_folder):
    for group in ['HOA', 'PD']:
        table_path_nap = os.path.join(data_folder, "parkinsons4", "nap", group, "table.csv")
        table_nap = pd.read_csv(table_path_nap)
        table_path_night = os.path.join(data_folder, "parkinsons4", "night", group, "table.csv")
        table_night = pd.read_csv(table_path_night)

        nap_ids = set(table_nap['record_id'])
        night_ids = set(table_night['record_id'])

        both_ids = np.array(list(nap_ids & night_ids))
        nap_only_ids = np.array(list(nap_ids - night_ids))
        night_only_ids = np.array(list(night_ids - nap_ids))

        record_ids = [both_ids, nap_only_ids, night_only_ids]

        train_subjects, valid_subjects, test_subjects = [], [], []

        np.random.seed(42)

        for ids in record_ids:
            n_total = len(ids)
            n_train = int(n_total * 0.7)
            n_test = int(n_total * 0.15)

            train_ids = np.random.choice(ids, size=n_train, replace=False)
            remaining_ids = np.setdiff1d(ids, train_ids, assume_unique=True)

            test_ids = np.random.choice(remaining_ids, size=n_test, replace=False)
            valid_ids = np.setdiff1d(remaining_ids, test_ids, assume_unique=True)

            train_subjects.extend(train_ids)
            valid_subjects.extend(valid_ids)
            test_subjects.extend(test_ids)

        
        train_subjects = np.array(train_subjects).reshape(1, -1)
        valid_subjects = np.array(valid_subjects).reshape(1, -1)
        test_subjects = np.array(test_subjects).reshape(1, -1)

        fold = 0
        table_nap.loc[
            table_nap["record_id"].isin(train_subjects[fold]), f"fold_{fold}"
        ] = "train"
        table_nap.loc[
            table_nap["record_id"].isin(valid_subjects[fold]), f"fold_{fold}"
        ] = "valid"
        table_nap.loc[table_nap["record_id"].isin(test_subjects[fold]), f"fold_{fold}"] = (
            "test"
        )

        table_night.loc[
            table_night["record_id"].isin(train_subjects[fold]), f"fold_{fold}"
        ] = "train"
        table_night.loc[
            table_night["record_id"].isin(valid_subjects[fold]), f"fold_{fold}"
        ] = "valid"
        table_night.loc[table_night["record_id"].isin(test_subjects[fold]), f"fold_{fold}"] = (
            "test"
        )

        logger.info("Saving the tables ...")
        table_nap.to_csv(table_path_nap)
        table_night.to_csv(table_path_night)
        

if __name__ == "__main__":

    ParkinsonsPreprocessor(data_folder="/home/coder/sleep/sleep-data/", night= True, healthy=True).run()
    ParkinsonsPreprocessor(data_folder="/home/coder/sleep/sleep-data/", night= True, healthy=False).run()
    ParkinsonsPreprocessor(data_folder="/home/coder/sleep/sleep-data/", night= False, healthy=True).run()
    ParkinsonsPreprocessor(data_folder="/home/coder/sleep/sleep-data/", night= False, healthy=False).run()

    global_split(data_folder="/home/coder/sleep/sleep-data/")