import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, resample

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing

from physioex.preprocess.utils.sleepdata import process_sleepdata_file

from physioex.preprocess.utils.sleepdata import get_channel_from_available, get_channels, read_channel_signal

POSSIBLE_EEG_CHANNELS = [
    ("EEG C4", "EEG A1"),
    ("EEG C4", "EEG A2"),
    ('EEG OPPEEG-C4', 'EEG OPPEEG-A1')
]

POSSIBLE_EOG_CHANNELS = [
    ("EOG LIO", "EOG RSO"),
    ("EOG OPPEEG-LIO", "EOG OPPEEG-RSO")
]

POSSIBLE_EMG_CHANNELS = [
    "EMG EMG CHIN",
    "EMG OPPEEG-EMG C"
]

def read_edf(edf_path, csv_path, ns2_path=None):
    stages_map = { 
        "Wake": 0,
        "N1": 1,
        "N2": 2, 
        "N3": 3, 
        "REM": 4
    }

    df = pd.read_csv(csv_path, sep=';')

    fs=100
    epoch_second=30
    epoch_microsecond = epoch_second*10**6

    stages = []
    stopping = None
    start_idx = -1
    stop_idx = -1

    for idx, row in df.iterrows():
        s = row['Subtype']
        if s in stages_map.keys():
            if start_idx == -1:
                start_idx = idx
            stop_idx = idx

            starting = float(row['Start time relative (total µs)'])

            if stopping:
                if starting != stopping:
                    length = int((starting-stopping)//epoch_microsecond)
                    if length < 0:
                        print('Overlapping labels, trusting start time over duration')
                        stages = stages[:length]
                    else:
                        stages.extend([-1]*length)
            
            duration = float(row['Duration (total µs)'])
            length = int(duration//epoch_microsecond)
            label = stages_map[s]
            stages.extend([label]*length)
            
            stopping = starting + duration


    scores_start = int(df['Start time relative (total µs)'].iloc[start_idx])*10**(-6)
    scores_end = int(df['Start time relative (total µs)'].iloc[stop_idx])*10**(-6) + int(df['Duration (total µs)'].iloc[stop_idx])*10**(-6)

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

    # Subj001 Night001 file has incomplete last epoch
    if edf_path == '/home/coder/sleep/sleep-data/intracranial/Intracranial_data/EEG/Subj001/Night001/Subj001_Night001_deidentified.edf':
        stages[-1] = -1

    # find the epochs associated with stages < 0 or >= 5
    stages = np.array(stages)
    invalid_epochs = np.where(np.logical_or(stages < 0, stages >= 5))[0]

    # remove the invalid epochs
    stages = np.delete(stages, invalid_epochs)
    signal = np.delete(signal, invalid_epochs, axis=0)

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

class IntracranialPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
        scalp: bool = True
    ):
        
        scalp_str = "scalp" if scalp else "scalp_intra"
        dataset_name = f"intracranial/{scalp_str}"

        signal_shape = [3, 3000] if scalp else [3, 3000] # ADAPT TO INCLUDE INTRA CHANNELS

        super().__init__(
            dataset_name=dataset_name,
            signal_shape=signal_shape,
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

        self.scalp = scalp

        self.root_folder = os.path.join(self.data_folder, "intracranial")



    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # this method should be provided by the user
        # the method should return a list containing the path of each subject record
        # each path is needed to be passed as argument to the function read_subject_record(self, record)

        table_path = os.path.join(self.root_folder, "Intracranial_data", "Overview.csv")
        self.demographics_table = pd.read_csv(table_path, sep=';')

        table = self.demographics_table.copy()[["SubjID", "Night"]]
        table = table.sort_values(by="SubjID").reset_index(drop=True)
        
        records = []
        record_dict = {}
        count = 0
        
        subjects = set(list(table['SubjID']))
        for subj in subjects:
            records_dir = os.path.join(self.root_folder, "Intracranial_data", "EEG", subj)
            # check if the directory exists if not continue
            if not os.path.exists(records_dir):
                logger.warning(f"Record directory {records_dir} does not exist. Skipping subject {subj}.")
                continue

            records.append(records_dir)
            record_dict[count] = subj
            count += 1

        self.record_dict = record_dict

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        print(record)

        nights = [n for n in os.listdir(record)if n.startswith('Night')]

        signal = []
        labels = []

        # Concatenate night 1 and 2
        for night in nights:
            print(night)
            files = os.listdir(os.path.join(record, night))
            edf_path = [f for f in files if f.endswith(('.edf', '.EDF'))][0]
            csv_path = [f for f in files if f.endswith('_Default.csv')][0]
            if self.scalp:
                ns2_path = None
                signal_tmp, labels_tmp = read_edf(os.path.join(record, night, edf_path), os.path.join(record, night, csv_path), None)
            else:
                ns2_path = [f for f in files if f.endswith('.ns2')]  
                signal_tmp, labels_tmp = read_edf(os.path.join(record, night, edf_path), os.path.join(record, night, csv_path), os.path.join(record, night, ns2_path))

            signal.extend(signal_tmp)
            labels.extend(labels_tmp)

        return np.array(signal), np.array(labels)

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:

        table["record_id"] = table["subject_id"].map(self.record_dict)
        
        return table
    
    @logger.catch
    def get_sets(self):
        """
        Performs LOOCV.
        """

        # Lists to store train/val/test splits for all folds
        all_train_folds, all_val_folds, all_test_folds = [], [], []

        np.random.seed(42)

        table = self.table.copy()
        subjects = np.array(table['subject_id'])
        shuffled_subjects = np.random.permutation(subjects) 
        k = len(subjects)

        for fold in range(k):
            remaining_subjects = list(shuffled_subjects)
            idx = shuffled_subjects[fold]
            test = [idx]
            remaining_subjects.remove(idx)
            idx = shuffled_subjects[(fold+1)%k]
            val = [idx]
            remaining_subjects.remove(idx)
            train = remaining_subjects

            all_train_folds.append(train)
            all_val_folds.append(val)
            all_test_folds.append(test)

        return all_train_folds, all_val_folds, all_test_folds


if __name__ == "__main__":

    IntracranialPreprocessor(data_folder="/home/coder/sleep/sleep-data/", scalp=True).run()
