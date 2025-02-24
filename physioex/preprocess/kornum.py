import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing_mouse

from physioex.preprocess.utils.mousedata import process_sleepdata_file


class KornumPreprocessor(Preprocessor):

    def __init__(
        self,
        preprocessors_name: List[str] = ["xsleepnet_mouse"],
        preprocessors=[xsleepnet_preprocessing_mouse],
        preprocessor_shape=[[3, 17, 129]],
        data_folder: str = None,
    ):

        super().__init__(
            dataset_name="kornum",
            signal_shape=[3, 512],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> np.ndarray:
        """
        Finds all .edf files in the data folder and extracts the subject ID from the file name.

        Returns:
            np.ndarray: An array of unique subject IDs.
        """

        scorer_list = []
        id_list = []
        location_list = []
        
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".edf"):
                    scorer_list.append(root.split('/')[-2][11:])
                    id_list.append(file.split('-')[0].upper())
                    location_list.append(os.path.join(root, file))
        
        self.database = pd.DataFrame({'scorer': scorer_list, 'id': id_list, 'location': location_list})

        return self.database.id.unique()

    
    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        """
        Reads all recordings belonging to 'record', processes and concatenates them.
        
        Args:
            record (str): The identifier of the subject to be read.
            should be skipped, the function returns (None, None).

        Returns:
            Tuple[np.array, np.array]: A tuple containing the signal and labels with shapes
            [n_windows, n_channels, n_timestamps] and [n_windows], respectively. If the record
            should be skipped, the function should return None, None.
        """
                
        subject_recordings = self.database[self.database['id'] == record]['location']
        
        recordings = []
        labels = []
        for rec in subject_recordings:
            tsv_file = rec.replace(".edf", ".tsv")
            tsv_file = tsv_file.replace("EDF", "tsv")
            
            signal, stages = process_sleepdata_file(rec, tsv_file)
                            
            recordings.append(signal)
            labels.append(stages)
        
        signal = np.concatenate(recordings, axis=0)
        stages = np.concatenate(labels, axis=0)
        
        return signal, stages    
        
    def get_sets(self) -> Tuple[np.array, np.array, np.array]:
        """
        Splits subjects into Train, Validation, and Test sets using a greedy allocation strategy.

        The function assigns each subject to the set that has the lowest proportion filled 
        relative to its target allocation ratio. This ensures a correct distribution 
        of sleep epochs according to the predefined ratios, while keeping subjects segregated.

        Returns:
            Tuple[np.array, np.array, np.array]: A tuple containing the train, validation, and test subjects.
        """

        np.random.seed(42)
        
        durations = self.table['num_windows'].values.tolist()
        subjects = self.table['subject_id'].values.tolist()
        
        total_duration = sum(durations)
        train_ratio = 0.7  
        val_ratio = 0.15
        test_ratio = 1 - train_ratio - val_ratio 
        
        indices = np.argsort(-np.array(durations))  # Sort by descending duration
        train, val, test = [], [], []
        train_dur, val_dur, test_dur = 0, 0, 0
        for i in indices:
            subject = subjects[i]
            dur = durations[i]
            
            # Compute current filling proportions relative to target
            train_fill = train_dur / (train_ratio * total_duration)
            val_fill = val_dur / (val_ratio * total_duration)
            test_fill = test_dur / (test_ratio * total_duration)

            # Assign the subject to the least filled set
            if train_fill <= val_fill and train_fill <= test_fill:
                train.append(subject)
                train_dur += dur
            elif val_fill <= train_fill and val_fill <= test_fill:
                val.append(subject)
                val_dur += dur
            else:
                test.append(subject)
                test_dur += dur
        
        return (np.array(train).reshape(1, -1),
                np.array(val).reshape(1, -1),
                np.array(test).reshape(1, -1),
        )


if __name__ == "__main__":

    p = KornumPreprocessor(data_folder="/Users/tlj258/Library/CloudStorage/OneDrive-UniversityofCopenhagen/Documents/THESIS_DATA/EEGdata_cleaned_physioex")

    p.run()
