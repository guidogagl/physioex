import os
import pickle
import stat
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from physioex.preprocess.utils.signal import online_variance

from physioex.utils import get_data_folder, set_data_folder

import h5py

class Preprocessor:

    def __init__(
        self,
        dataset_name: str,
        signal_shape: List[int],
        preprocessors_name: List[str],
        preprocessors: List[Callable],
        preprocessors_shape: List[List[int]],
        data_folder: str = None,
        batch_size: int = 1000,
    ):

        assert (
            len(signal_shape) == 2
        ), "ERR: signal_shape should be a list of 2 elements n_channels, n_timestamps"

        assert (
            len(preprocessors_name) == len(preprocessors) == len(preprocessors_shape)
        ), "ERR: lists preprocessors_name, preprocessors e preprocessors_shape should match first dimension"

        
        self.data_folder = (
            get_data_folder() if data_folder is None else set_data_folder(data_folder)
        )

        self.data_folder = data_folder

        self.dataset_folder = os.path.join(self.data_folder, dataset_name)
        os.makedirs(self.dataset_folder, exist_ok=True)

        self.signal_shape = signal_shape
        self.preprocessors_name = preprocessors_name
        self.preprocessors = preprocessors
        self.preprocessors_shape = preprocessors_shape
        self.batch_size = batch_size

    @logger.catch
    def download_dataset(self) -> None:
        # this method should be provided by the user
        # the method should take care of checking if the dataset is already on disk
        pass

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # this method should be provided by the user
        # the method should return a list containing the path of each subject record
        # each path is needed to be passed as argument to the function read_subject_record(self, record)

        pass

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        # this method should be provided by the user
        # the method should return a tuple signal, label with shape [ n_windows, n_channels, n_timestamps ], [ n_windows ]
        # if the record should be skipped the function should return None, None
        pass

    @logger.catch
    def customize_table(self, table) -> pd.DataFrame:
        # this method should be provided by the user
        # the method should return a customized version of the dataset table before saving it
        return table

    @logger.catch
    def get_sets(self) -> Tuple[List, List, List]:
        # this method should be provided by the user
        # the method should return the train valid and test subjects,

        np.random.seed(42)

        table = self.table.copy()

        train_subjects = np.random.choice(
            table["subject_id"], size=int(table.shape[0] * 0.7), replace=False
        )
        valid_subjects = np.setdiff1d(
            table["subject_id"], train_subjects, assume_unique=True
        )
        test_subjects = np.random.choice(
            valid_subjects, size=int(table.shape[0] * 0.15), replace=False
        )
        valid_subjects = np.setdiff1d(valid_subjects, test_subjects, assume_unique=True)

        return (
            train_subjects.reshape(1, -1),
            valid_subjects.reshape(1, -1),
            test_subjects.reshape(1, -1),
        )

    def run(self):
        logger.info("Downloading the dataset if needed ...")
        self.download_dataset()

        labels_path = os.path.join(self.dataset_folder, "labels/")
        raw_path = os.path.join(self.dataset_folder, "raw/")
        prep_path = [
            os.path.join(self.dataset_folder, name + "/")
            for name in self.preprocessors_name
        ]

        # create the folders if they do not exist
        Path(labels_path).mkdir(parents=True, exist_ok=True)
        Path(raw_path).mkdir(parents=True, exist_ok=True)
        for path in prep_path:
            Path(path).mkdir(parents=True, exist_ok=True)

        logger.info("Fetching the dataset ...")
        
        index_to_subject = []
        index_to_index = []
        
        raw_var = online_variance().set_shape(self.signal_shape)
        prep_var = [
            online_variance().set_shape(self.preprocessors_shape[i])
            for i in range(len(self.preprocessors_name))
        ]

        h5_file_path = os.path.join(self.dataset_folder, "dataset.h5")
        
        with h5py.File(h5_file_path, 'w') as h5_file:
        
            for subject_id, subject_records in enumerate(tqdm(self.get_subjects_records())):
                signal, labels = self.read_subject_record(subject_records)

                if signal is None and labels is None:
                    logger.warning(
                        f"subject record - {subject_records} - is being discarded"
                    )
                    continue
                
                # compute the online mean and variance
                raw_var.add(signal)

                num_windows = labels.shape[0]

                index_to_subject += [subject_id] * num_windows
                index_to_index += list(range(num_windows))

                group = h5_file.create_group(str(subject_id))
                group.create_dataset("raw", data=signal.astype(np.float32), dtype=np.float32)
                group.create_dataset("labels", data=labels.astype(np.int16), dtype=np.int16)

                for i, prep_name in enumerate( self.preprocessors_name ):
                    p_signal = self.preprocessors[i](signal)
                    
                    group.create_dataset(prep_name, data=p_signal.astype(np.float32), dtype=np.float32)
                    
                    prep_var[i].add(p_signal)
                
            index_to_subject = np.array(index_to_subject).astype(np.uint16)
            index_to_index = np.array(index_to_index).astype(np.uint32)    

            h5_file.create_dataset("index_to_subject", data=index_to_subject, dtype=np.uint16)
            h5_file.create_dataset("index_to_index", data=index_to_index, dtype=np.uint32)
            
        self.table = table

        logger.info("Computing splitting parameters ...")

        train_subjects, valid_subjects, test_subjects = self.get_sets()

        # assert the same number of folds in each split
        assert (
            len(train_subjects) == len(valid_subjects) == len(test_subjects)
        ), "ERR: number of folds in each split should be the same"

        for fold in range(len(train_subjects)):
            # add a "fold_{fold}" column to the table with the split of the subject ( train, valid or test )
            table.loc[
                table["subject_id"].isin(train_subjects[fold]), f"fold_{fold}"
            ] = "train"
            table.loc[
                table["subject_id"].isin(valid_subjects[fold]), f"fold_{fold}"
            ] = "valid"
            table.loc[table["subject_id"].isin(test_subjects[fold]), f"fold_{fold}"] = (
                "test"
            )

        logger.info("Saving the table ...")
        table_path = os.path.join(self.dataset_folder, "table.csv")
        table.to_csv(table_path)

        logger.info("Computing scaling parameters ...")

        raw_mean, raw_std = raw_var.compute()
        print(raw_mean.shape, raw_std.shape)

        scaling_path = os.path.join(raw_path, "scaling.npz")
        np.savez(scaling_path, mean=raw_mean, std=raw_std)

        for i, name in enumerate(self.preprocessors_name):
            scaling_path = os.path.join(prep_path[i], "scaling.npz")
            p_mean, p_std = prep_var[i].compute()

            print(p_mean.shape, p_std.shape)

            np.savez(scaling_path, mean=p_mean, std=p_std)
