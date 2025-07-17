import os
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import datalad.api as dl
import openneuro as on
from scipy.signal import filtfilt, firwin, resample

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing_mouse

from physioex.preprocess.utils.mousedata import get_channels, read_channel_signal



def process_recording(edf_path, tsv_path):

    fs = 100
    epoch_second = 4

    available_channels = get_channels(edf_path)
    
    try:
        stages = pd.read_csv(tsv_path, sep='\t')['stage'].values
        stages = stages - 1
    except Exception as e:
        print(f"Error reading file: {tsv_path}")
        print(f"skipping subject")
        return None, None
    
    eeg_candidates = [ch for ch in available_channels if 'EEG' in ch.upper()]
    eeg_channel = random.choice(eeg_candidates) if eeg_candidates else None

    eeg1, old_fs = read_channel_signal(edf_path, eeg_channel)

    # Parametri
    Nfir = 100

    # Creazione del filtro FIR bandpass
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

    # Applicazione del filtro al segnale EEG
    eeg1 = filtfilt(b_band, 1, eeg1)

    if fs != old_fs:
        eeg1 = resample(eeg1, int(len(eeg1) * fs / old_fs))
     
    eeg2 = eeg1.copy() # only working with one EEG channel for now, but filling eeg2 for shape coherence with other datasets
    
    emg_candidates = [ch for ch in available_channels if 'EMG' in ch.upper()]
    emg_channel = random.choice(eeg_candidates) if emg_candidates else None
    if emg_channel is None:
        print(f"Error: no EMG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        emg, old_fs = read_channel_signal(edf_path, emg_channel)

    # filtering and resampling
    b_band = firwin(Nfir + 1, 10, pass_zero=False, fs=old_fs)
    emg = filtfilt(b_band, 1, emg)

    if fs != old_fs:
        emg = resample(emg, int(len(emg) * fs / old_fs))

    expected_epochs = len(eeg1) // (epoch_second * fs)

    # checking coherence of the signals with stages
    if expected_epochs > len(stages):
        expected_epochs = len(stages)
    else:
        stages = stages[:expected_epochs]
    total_samples = expected_epochs * epoch_second * fs
    eeg1 = eeg1[:total_samples]
    eeg2 = eeg2[:total_samples]
    emg = emg[:total_samples]
    stages = np.array(stages)
    # print stages distribution
    # print(f'Stages distribution: {np.bincount(stages)}')

    # buffer the signals into epochs
    signal = np.array([eeg1, eeg2, emg])
    signal = np.transpose(signal).reshape(expected_epochs, epoch_second * fs, 3)

    # find the epochs associated with stages < 0 or > 2
    invalid_epochs = np.where(np.logical_or(stages < 0, stages > 2))[0]

    # remove the invalid epochs
    stages = np.delete(stages, invalid_epochs)
    signal = np.delete(signal, invalid_epochs, axis=0)
    
    signal = np.transpose(signal, (0, 2, 1))

    return signal.astype(np.float32), stages.astype(int)


class RecordingsIterator:
    '''
    So different recordings from same subject are saved separately.
    
    '''
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self._iter = data.iterrows()  # (index, row)
            self.len = len(data)
        else:
            raise TypeError("Unsupported data type")

    def __iter__(self):
        return self

    def __next__(self):
        _, row = next(self._iter)
        return row
    
    def __len__(self):
        return self.len


class MSSVPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet_mouse"],
        preprocessors=[xsleepnet_preprocessing_mouse],
        preprocessor_shape=[[3, 17, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="mssv",
            signal_shape=[3, 400],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )
        
        self.source_dataset = os.path.join(self.dataset_folder, 'mssv_openneuro')
        
        self.split_subjects_table = None
        self._iterator = None
        
    @logger.catch
    def download_dataset(self) -> None:
        """
        Downloads the dataset if it is not already present on disk.

        """
        # pass

        # if not os.listdir(self.source_dataset):
        #     raise NotImplementedError(
        #         "❌ Automatic download of MSSV is not supported yet. "
        #         f"Please download the dataset manually from https://openneuro.org/datasets/ds006366/versions/1.0.0/download# in the directory {self.source_dataset}"
        #     )

        if not os.path.exists(self.source_dataset):

            os.makedirs(self.source_dataset)

        print(
            "The openneuro downloader is very unstable and might crash before finishing. "
            "Rerun the script as many times as needed to resume the download until "
            "the download finishes and the actual preprocessing starts."
        )
        print("")

        on.download(dataset='ds006366', target_dir=self.source_dataset)


    @logger.catch
    def get_subjects_records(self) -> np.ndarray:
        """
        Finds all .edf files in the data folder and extracts the subject ID from the file name.

        Returns:
            np.ndarray: An array of unique subject IDs.
        """
        
        if self.split_subjects_table is None:

            participants_path = os.path.join(self.source_dataset, 'participants.tsv')
            self.participants = pd.read_csv(participants_path, sep='\t')
            
            split_subjects_rows = []    
                
            for _, row in self.participants.iterrows():
                subj_folder = os.path.join(self.source_dataset, row['participant_id'])
                
                eeg_files = [f for f in os.listdir(subj_folder + '/eeg') if f.endswith('eeg.edf')]
                
                for f in eeg_files:
                    run = f.split('_')[-2]
                    
                    split_subjects_rows.append({
                        'edf_path': f,
                        'real_subject': row['participant_id'],
                        'run': run,
                        'lab': row['lab']
                    })
                    
            self.split_subjects_table = pd.DataFrame(split_subjects_rows)
        
        self._iterator = RecordingsIterator(self.split_subjects_table)
                  
        return self._iterator

    
    @logger.catch
    def read_subject_record(self, record: pd.Series) -> Tuple[np.array, np.array]:
        """
        Reads all recordings belonging to 'record', processes and concatenates them.

        Args:
            record (pd.Series): The row representing the subject's recording.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the signal and labels with shapes
            [n_windows, n_channels, n_timestamps] and [n_windows], respectively. If the record
            should be skipped, the function should return None, None.
        """
                
        subject_folder = os.path.join(self.source_dataset, record['real_subject'])
        
        edf_path = os.path.join(subject_folder, 'eeg', record['edf_path'])
        tsv_path = os.path.join(subject_folder, 'eeg', record['edf_path'].replace('_eeg.edf', '_events.tsv'))
        
        signal, stages = process_recording(edf_path, tsv_path)
                
        return signal, stages
    
    def customize_table(self, table) -> pd.DataFrame:
        """
        Customizes the dataset table before saving it.


        Parameters:
            table (pd.DataFrame): The dataset table to be customized.

        Returns:
            pd.DataFrame: The customized dataset table.
        """
        
        return table.join(self.split_subjects_table[['real_subject', 'run', 'lab']])
    
    
    def get_sets(self, k=4) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        """
        Performs K-Fold splitting using a greedy allocation strategy,
        stratified by 'lab' to ensure balanced lab distribution in each fold.
        
        The greedy strategy also assigns each subject to the set that has the lowest proportion filled 
        relative to its target allocation ratio. This is done because some of the mice have much
        more epochs than others. This ensures a correct distribution of sleep epochs according to
        the predefined ratios, while keeping mice segregated.

        Args:
            k (int): Number of folds.

        Returns:
            Tuple[List[np.array], List[np.array], List[np.array]]:
            Lists of train, validation, and test sets for each fold.
        """

        # 1. Aggregate per subject
        subject_groups = self.table.groupby('real_subject')
        subject_ids = np.array(list(subject_groups.groups.keys()))
        subject_durations = subject_groups['num_windows'].sum().values
        subject_labs = subject_groups['lab'].first().values  # assumes consistent lab per subject

        total_duration = subject_durations.sum()
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 1 - train_ratio - val_ratio

        used_test_subjects = set()

        all_train_folds, all_val_folds, all_test_folds = [], [], []

        np.random.seed(42)
        unique_labs = np.unique(subject_labs)

        for fold in range(k):
            train_subjects, val_subjects, test_subjects = [], [], []

            for lab in unique_labs:
                lab_mask = subject_labs == lab
                lab_subject_ids = subject_ids[lab_mask]
                lab_durations = subject_durations[lab_mask]

                # Shuffle lab-specific subjects
                perm = np.random.permutation(len(lab_subject_ids))
                lab_subject_ids = lab_subject_ids[perm]
                lab_durations = lab_durations[perm]

                lab_total_duration = lab_durations.sum()
                lab_test_dur, lab_val_dur = 0, 0
                lab_test, lab_val, lab_train = [], [], []

                for subj, dur in zip(lab_subject_ids, lab_durations):
                    if subj in used_test_subjects:
                        continue
                    if lab_test_dur + dur <= test_ratio * lab_total_duration:
                        lab_test.append(subj)
                        lab_test_dur += dur
                        used_test_subjects.add(subj)

                remaining = [s for s in lab_subject_ids if s not in lab_test]

                for subj in remaining:
                    idx = np.where(subject_ids == subj)[0][0]
                    dur = subject_durations[idx]
                    if lab_val_dur + dur <= val_ratio * lab_total_duration:
                        lab_val.append(subj)
                        lab_val_dur += dur

                lab_train = [s for s in remaining if s not in lab_val]

                train_subjects.extend(lab_train)
                val_subjects.extend(lab_val)
                test_subjects.extend(lab_test)

            # Convert to arrays
            train_subjects = np.array(train_subjects)
            val_subjects = np.array(val_subjects)
            test_subjects = np.array(test_subjects)

            # Map back to full table: get all recordings for each subject
            train = self.table[self.table['real_subject'].isin(train_subjects)]
            val = self.table[self.table['real_subject'].isin(val_subjects)]
            test = self.table[self.table['real_subject'].isin(test_subjects)]

            # Compute durations
            train_dur = train['num_windows'].sum()
            val_dur = val['num_windows'].sum()
            test_dur = test['num_windows'].sum()

            train_prop = train_dur / total_duration
            val_prop = val_dur / total_duration
            test_prop = test_dur / total_duration

            # Store row indices corresponding to each subject set
            train_indices = train.index.values
            val_indices = val.index.values
            test_indices = test.index.values

            all_train_folds.append(train_indices)
            all_val_folds.append(val_indices)
            all_test_folds.append(test_indices)

            print(f"\n===== Fold {fold + 1} =====")
            print(f"Train Subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Validation Subjects ({len(val_subjects)}): {val_subjects}")
            print(f"Test Subjects ({len(test_subjects)}): {test_subjects}")
            print(f"Epoch Distribution: Train {train_prop:.2%}, Val {val_prop:.2%}, Test {test_prop:.2%}")

            # Optional: per-lab distribution check
            def lab_stats(name, df):
                return f"{name} lab distribution:\n{df.groupby('lab')['num_windows'].sum() / df['num_windows'].sum()}\n"

            print(lab_stats("Train", train))
            print(lab_stats("Validation", val))
            print(lab_stats("Test", test))

        return all_train_folds, all_val_folds, all_test_folds



if __name__ == "__main__":

    p = MSSVPreprocessor(data_folder="/home/coder/sleep/sleep-data/")

    p.run()
