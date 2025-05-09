import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

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
            signal_shape=[3, 400],
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
        
        for root, _, files in os.walk(self.dataset_folder):
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
        
    def get_sets(self, k=4) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        """    
        Performs K-Fold splitting using a greedy allocation strategy.
        
        The greedy strategy assigns each subject to the set that has the lowest proportion filled 
        relative to its target allocation ratio. This is done because some of the mice have much
        more epochs than others. This ensures a correct distribution of sleep epochs according to
        the predefined ratios, while keeping mice segregated.

        Args:
            k (int): Number of folds.

        Returns:
            Tuple[List[np.array], List[np.array], List[np.array]]: 
            Lists of train, validation, and test sets for each fold.
        """

        durations = np.array(self.table['num_windows'].values)  # Convert durations to NumPy array
        subjects = np.array(self.table['subject_id'])  # Convert to NumPy array for indexing
        
        total_duration = sum(durations)
        train_ratio = 0.7  
        val_ratio = 0.15
        test_ratio = 1 - train_ratio - val_ratio 

        # Keep track of which subjects have been used in test/validation sets
        used_test_subjects = set()
        used_val_subjects = set()

        # Lists to store train/val/test splits for all folds
        all_train_folds, all_val_folds, all_test_folds = [], [], []

        np.random.seed(42)  # Ensure reproducibility
        shuffled_subjects = np.random.permutation(subjects)  # Randomize subject order

        for fold in range(k):
            remaining_subjects = list(shuffled_subjects)  # Copy subject list for current fold

            # Assign subjects to test set incrementally until test_ratio is reached
            test, test_dur = [], 0
            np.random.shuffle(remaining_subjects)
            for subject in remaining_subjects[:]:  # Iterate over a copy
                if subject in used_test_subjects:
                    continue  # Skip if subject was already in a test set before

                subject_idx = np.where(subjects == subject)[0][0]
                subject_dur = durations[subject_idx]

                if (test_dur + subject_dur) / total_duration <= test_ratio:
                    test.append(subject)
                    test_dur += subject_dur
                    used_test_subjects.add(subject)  # Mark as used
                    remaining_subjects.remove(subject)

            # Assign subjects to validation set incrementally until val_ratio is reached
            val, val_dur = [], 0
            np.random.shuffle(remaining_subjects)
            for subject in remaining_subjects[:]:
                if subject in used_val_subjects:
                    continue  # Skip if subject was already in a validation set before

                subject_idx = np.where(subjects == subject)[0][0]
                subject_dur = durations[subject_idx]

                if (val_dur + subject_dur) / total_duration <= val_ratio:
                    val.append(subject)
                    val_dur += subject_dur
                    used_val_subjects.add(subject)  # Mark as used
                    remaining_subjects.remove(subject)

            # Assign the remaining subjects to the training set
            train = remaining_subjects

            # Convert to NumPy arrays
            train, val, test = np.array(train), np.array(val), np.array(test)

            # Compute durations properly
            train_dur = sum(durations[np.isin(subjects, train)]) if len(train) > 0 else 0
            test_dur = sum(durations[np.isin(subjects, test)]) if len(test) > 0 else 0
            val_dur = sum(durations[np.isin(subjects, val)]) if len(val) > 0 else 0

            train_prop = train_dur / total_duration
            val_prop = val_dur / total_duration
            test_prop = test_dur / total_duration

            # Append this fold's split
            all_train_folds.append(train)
            all_val_folds.append(val)
            all_test_folds.append(test)

            # Print fold details
            print(f"\n===== Fold {fold + 1} =====")
            print(f"Train Subjects ({len(train)}): {train}")
            print(f"Validation Subjects ({len(val)}): {val}")
            print(f"Test Subjects ({len(test)}): {test}")
            print(f"Epoch Distribution: Train {train_prop:.2%}, Val {val_prop:.2%}, Test {test_prop:.2%}")

        return all_train_folds, all_val_folds, all_test_folds




if __name__ == "__main__":

    p = KornumPreprocessor(data_folder="/esat/biomeddata/ggagliar/")

    p.run()
