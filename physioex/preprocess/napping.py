import os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve
import mne

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin, resample, resample_poly

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing

from physioex.preprocess.utils.sleepdata import process_sleepdata_file

from physioex.preprocess.utils.sleepdata import get_channel_from_available, get_channels, read_channel_signal

POSSIBLE_EEG_CHANNELS = [
    "EEG C4-A1"
]

POSSIBLE_EOG_CHANNELS = [
    ("EOG LEFT", "EOG RIGHT")
]

POSSIBLE_EMG_CHANNELS = [
    "EMG CHIN"
]

def read_edf(rec_path, events_path, night):
    stages_map = { 
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2, 
        "Sleep stage 3": 3, 
        "Sleep stage 4": 3,
        "Sleep stage R": 4
    }
    fs=100
    epoch_second=30

    # Get annotations
    events_data = mne.io.read_raw_edf(events_path, preload=False)
    annotations = [
        (stages_map[ann['description']], ann['onset'], ann['duration'])
        for ann in events_data.annotations
        if ann['description'] in stages_map
        ]

    start = annotations[0][1]

    # Get one annotation per second
    stages_per_second = []
    stopping = start
    for i, (descr, onset, dur) in enumerate(annotations):
        if onset != stopping:
            length = int(onset-stopping)
            stages_per_second.extend([-1]*length)
        stages_per_second.extend([descr]*int(dur))
        stopping = onset+dur

            
    # Get one annotation per epoch
    stages = []
    for i in range(len(stages_per_second)//epoch_second): # convert annotations per second to annotations per epoch    
        epoch_annots = stages_per_second[i*epoch_second : (i+1)*epoch_second]
        unique_labels = set(epoch_annots)
        if len(unique_labels) == 1:
            stages.append(unique_labels.pop()) # add unique label
        else:
            stages.append(-1) # add invalid label if not unique

    if not night:
        start_sleep = next((i for i, x in enumerate(stages) if x != 0), -1)

        if start_sleep*epoch_second >= start+20*60 or start_sleep == -1: # if no sleep in the first 20 minutes, MSLT nap opportunity is stopped
            end_test = int((start+20*60)/epoch_second)
        else: # else, MSLT nap opportunity is stopped 15 minutes after sleep onset
            end_test = start_sleep + int(15*60/epoch_second)
        end_test = max(end_test, len(stages))
        stages = stages[:end_test] 

    available_channels = get_channels(rec_path)
    eeg_channel = get_channel_from_available(available_channels, POSSIBLE_EEG_CHANNELS)
    if eeg_channel is None:
        print(f"Error: no EEG channel found in {rec_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eeg, old_fs = read_channel_signal(rec_path, eeg_channel, start, n=len(stages)*epoch_second)
        
    # upsample for compatibility
    if old_fs < fs:
        eeg = resample_poly(eeg, fs/old_fs, 1)
        old_fs = fs

    # Creazione del filtro FIR bandpass
    Nfir = 500
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

    # Applicazione del filtro al segnale EEG
    eeg = filtfilt(b_band, 1, eeg)

    if fs != old_fs:
        eeg = resample(eeg, int(len(eeg) * fs / old_fs))

    eog_channel = get_channel_from_available(available_channels, POSSIBLE_EOG_CHANNELS)
    if eog_channel is None:
        print(f"Error: no EOG channel found in {rec_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eog, old_fs = read_channel_signal(rec_path, eog_channel, start, n=len(stages)*epoch_second)

    # upsample for compatibility
    if old_fs < fs:
        eog = resample_poly(eog, fs/old_fs, 1)
        old_fs = fs

    # filtering and resampling
    eog = filtfilt(b_band, 1, eog)

    if fs != old_fs:
        eog = resample(eog, int(len(eog) * fs / old_fs))

    emg_channel = get_channel_from_available(available_channels, POSSIBLE_EMG_CHANNELS)
    if emg_channel is None:
        print(f"Error: no EMG channel found in {rec_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        emg, old_fs = read_channel_signal(rec_path, emg_channel, start, n=len(stages)*epoch_second)

    # upsample for compatibility
    if old_fs < fs:
        emg = resample_poly(emg, fs/old_fs, 1)
        old_fs = fs

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

    if night:
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

class NappingPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
        psg: bool = True,
        group: str = None,
        mslt: int = 0
    ):

        psg_str = "psg" if psg else "mslt"+str(mslt)
        
        dataset_name = f"napping/{psg_str}/{group}"

        super().__init__(
            dataset_name=dataset_name,
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

        self.psg = psg
        self.group = group
        self.mslt = mslt

        self.root_folder = os.path.join(self.data_folder, "napping")



    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # this method should be provided by the user
        # the method should return a list containing the path of each subject record
        # each path is needed to be passed as argument to the function read_subject_record(self, record)

        table_path = os.path.join(self.root_folder, "Napping_data", "Napping_data", "demographics.csv")
        self.demographics_table = pd.read_csv(table_path, sep=';')
        
        table = self.demographics_table.copy()[["subj_id", "group"]]
        table = table.sort_values(by="subj_id").reset_index(drop=True)

        # now loop into the couples "subj_id" and "group" to create the records list
        records_list = []
        records_dict = {}
        count = 0
        for idx, row in table.iterrows():
            subject_id = row['subj_id']
            group = row['group']

            if self.group != group:
                continue
                
            records_name = os.path.join(self.root_folder, "Napping_data", "Napping_data", str(subject_id), str(subject_id) + "_PSG" if self.psg else str(subject_id) + "_MSLT"  + str(self.mslt))
            # check if the directory exists if not continue
            if not os.path.exists(records_name +'.EDF'):
                if os.path.exists(records_name +'.edf'):
                    extension = '.edf'
                else:
                    logger.warning(f"Record {records_name} does not exist. Skipping subject {subject_id}.")
                    continue
            else:
                extension = '.EDF'

            records_list.append((records_name, extension))
            records_dict[count] = subject_id
            count += 1

        self.records_dict = records_dict

        return records_list

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        rec_path = record[0] + record[1]
        events_path = record[0] + '_events.EDF'

        signal, labels = read_edf(rec_path, events_path, self.psg)
        
        return signal, labels
    
    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:

        table["record_id"] = table["subject_id"].map(self.records_dict)
        
        return table
    

def global_split(data_folder, groups):

    mslt_sessions = ['mslt1', 'mslt2', 'mslt3', 'mslt4', 'mslt5']

    for group in groups:
        subjects = []
        for mslt in mslt_sessions:
            table_path = os.path.join(data_folder, "napping", mslt, group, "table.csv")
            table = pd.read_csv(table_path)
            for subject_id in table['record_id']:
                subjects.append(subject_id)

        nap_ids = set(subjects)

        table_path_night = os.path.join(data_folder, "napping", "psg", group, "table.csv")
        table_night = pd.read_csv(table_path_night)
        night_ids = set(table_night['record_id'])

        both_ids = np.array(sorted(nap_ids & night_ids))
        nap_only_ids = np.array(sorted(nap_ids - night_ids))
        night_only_ids = np.array(sorted(night_ids - nap_ids))

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
        split_col = f"fold_{fold}"
        split_mapping = {
            "train": train_subjects[fold],
            "valid": valid_subjects[fold],
            "test": test_subjects[fold],
        }

        logger.info("Saving the tables ...")

        for mslt in mslt_sessions:
            table_path = os.path.join(data_folder, "napping", mslt, group, "table.csv")
            table = pd.read_csv(table_path)

            # Assign split label
            table[split_col] = "none"
            for split, ids in split_mapping.items():
                table.loc[table["record_id"].isin(ids), split_col] = split

            # Save the updated table back
            table.to_csv(table_path, index=False)

        for split, ids in split_mapping.items():
            table_night.loc[table_night["record_id"].isin(ids), split_col] = split
        table_night.to_csv(table_path_night, index=False)


if __name__ == "__main__":

    groups = ['control', 'NT1', 'NT2', 'IHS']

    for group in groups:
        NappingPreprocessor(data_folder="/home/coder/sleep/sleep-data/", psg= True, group=group).run()
        for mslt in [1, 2, 3, 4, 5]:
            NappingPreprocessor(data_folder="/home/coder/sleep/sleep-data/", psg= False, group=group, mslt=mslt).run()

    global_split("/home/coder/sleep/sleep-data/", groups)
