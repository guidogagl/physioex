import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import filtfilt, firwin, resample

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing_mouse

from physioex.preprocess.utils.mousedata import get_channels, get_channel_from_available, read_channel_signal, POSSIBLE_EEG1_CHANNELS, POSSIBLE_EEG2_CHANNELS, POSSIBLE_EMG_CHANNELS


class SleepyRatPreprocessor(Preprocessor):

    def __init__(
        self,
        cohorts: List[str] = ["A", "D"], # Only healthy mice, but the code supports all cohorts
        scorer: int = 2, # 1 or 2
        only_test: bool = True, # if True, all data is assigned to test set
        preprocessors_name: List[str] = ["xsleepnet_mouse"],
        preprocessors=[xsleepnet_preprocessing_mouse],
        preprocessor_shape=[[3, 17, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="sleepyrat",
            signal_shape=[3, 512],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )
        
        self.cohorts = [c.upper() for c in cohorts]
        self.scorer = scorer
        self.only_test = only_test


    @logger.catch
    def get_subjects_records(self) -> List[str]:
        """
        Finds all .edf files in the data folder and extracts the subject ID from the file name.

        Returns:
            np.ndarray: An array of unique subject IDs.
        """

        id_list = []
        location_list = []
        
        for cohort in self.cohorts:
            recordings_path = Path(self.dataset_folder) / "original_data" / f"Cohort{cohort}" / "recordings"
            for recording in recordings_path.glob("*.edf"):
                id_list.append(recording.stem)
                location_list.append(str(recording))
            
        self.database = pd.DataFrame({'id': id_list, 'location': location_list})

        return self.database.id.tolist()
    
    
    def process_scorings(self, scorings_path):
        df = pd.read_csv(scorings_path, header=None)

        # column names: {1, 2, 3, n, r, w}
        # 1=wake artifact, 2=NREM artifact, 3=REM artifact

        if self.scorer == 1:
            labels = df[1]
        elif self.scorer == 2:
            labels = df[2]

        # rename classes and convert class artifacts to unique artifact class
        def rename_class(row):
            if row == "w":
                return 0
            if row == "n":
                return 1
            if row == "r":
                return 2
            if (row != "w") & (row != "n") & (row != "r"):
                return 3

        stages = labels.apply(rename_class)

        return stages.to_list()
        
        
    def process_recording(self, edf_path, stages):
        '''
        mousedata:process_sleepdata_file, adapted for SleepyRat dataset
        '''
        
        fs = 128
        epoch_second = 4

        # get the file name of the absolute path filename without the extension
        name = os.path.basename(edf_path)
        name = os.path.splitext(name)[0]

        available_channels = get_channels(edf_path)
        
        eeg1_channel = get_channel_from_available(available_channels, POSSIBLE_EEG1_CHANNELS)

        if eeg1_channel is None:
            print(f"Error: no EEG1 channel found in {edf_path}")
            print(f"Available channels: {available_channels}")
            return None, None
        else:
            eeg1, old_fs = read_channel_signal(edf_path, eeg1_channel)

        # Parametri
        Nfir = 100

        # Creazione del filtro FIR bandpass
        b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

        # Applicazione del filtro al segnale EEG
        eeg1 = filtfilt(b_band, 1, eeg1)

        if fs != old_fs:
            eeg1 = resample(eeg1, int(len(eeg1) * fs / old_fs))

        eeg2_channel = get_channel_from_available(available_channels, POSSIBLE_EEG2_CHANNELS)
        if eeg2_channel is None:
            print(f"Error: no EEG2 channel found in {edf_path}")
            print(f"Available channels: {available_channels}")
            return None, None
        else:
            eeg2, old_fs = read_channel_signal(edf_path, eeg2_channel)

        # filtering and resampling
        eeg2 = filtfilt(b_band, 1, eeg2)

        if fs != old_fs:
            eeg2 = resample(eeg2, int(len(eeg2) * fs / old_fs))

        emg_channel = get_channel_from_available(available_channels, POSSIBLE_EMG_CHANNELS)
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
        
        
    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        """
        Reads recording belonging to 'record'.
        
        Args:
            record (str): The identifier of the subject to be read.

        Returns:
            Tuple[np.array, np.array]: A tuple containing the signal and labels with shapes
            [n_windows, n_channels, n_timestamps] and [n_windows], respectively.
        """
                
        recording_path = self.database[self.database['id'] == record]['location'].iloc[0]
        
        scorings_path = recording_path.replace("recordings", "scorings")
        scorings_path = scorings_path.replace("edf", "csv")
        
        stages = self.process_scorings(scorings_path)
        signal, stages = self.process_recording(recording_path, stages)
                        
        return signal, stages    
        

    def get_sets(self) -> Tuple[List[np.array], List[np.array], List[np.array]]:

        np.random.seed(42)

        if self.only_test:
            return [[]], [[]], [self.table["subject_id"].tolist()]
        else:
            return super().get_sets()




if __name__ == "__main__":

    p = SleepyRatPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
