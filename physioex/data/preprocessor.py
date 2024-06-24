import os
import pickle
import stat
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, filtfilt, resample, spectrogram
from tqdm import tqdm

from physioex.data.constant import get_data_folder, set_data_folder


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def compute_mean_std_incremental(data, batch_size, shape):
    total_sq, total = np.zeros(shape), np.zeros(shape)
    n = 0

    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size].copy()
        batch = batch.astype(np.double)
        total += np.sum(batch)
        total_sq += np.sum(np.square(batch))
        n += batch_size

    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 0))

    return mean.astype(np.float32), std.astype(np.float32)


def chmod_recursive(path, mode):
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, mode)
        for filename in filenames:
            os.chmod(os.path.join(dirpath, filename), mode)


def xsleepnet_preprocessing(signals):

    num_windows, n_channels, n_timestamps = signals.shape

    # transform each signal into its spectrogram ( fast )
    # nfft 256, noverlap 1, win 2, fs 100, hamming window

    S = np.zeros((num_windows, n_channels, 29, 129)).astype(np.float32)

    for i in range(num_windows):
        for j in range(n_channels):

            _, _, Sxx = spectrogram(
                signals[i, j].astype(np.double).reshape(-1),
                fs=100,
                window="hamming",
                nperseg=200,
                noverlap=100,
                nfft=256,
            )

            # log_10 scale the spectrogram safely (using epsilon)
            Sxx = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

            Sxx = np.transpose(Sxx, (1, 0))

            S[i, j] = Sxx.astype(np.float32)

    return S


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
        pass

    @logger.catch
    def get_dataset_num_windows(self) -> int:
        # this method should be provided by the user
        # the method should return a int number representing the total number of windows in the dataset. For efficiency the best would be to precompute this value.

        records = self.get_subjects_records()
        num_windows = 0
        for record in tqdm(records):
            _, labels = self.read_subject_record(record)

            num_windows += labels.shape[0]

        return num_windows

    def run(self):
        logger.info("Downloading the dataset if needed ...")
        self.download_dataset()

        logger.info("Computing the total number of windows in the dataset ...")
        tot_num_windows = self.get_dataset_num_windows()

        labels_path = os.path.join(self.dataset_folder, "labels.dat")
        labels_mp = np.memmap(
            labels_path, dtype="int16", mode="w+", shape=(tot_num_windows)
        )

        raw_path = os.path.join(self.dataset_folder, "raw.dat")
        raw_mp = np.memmap(
            raw_path,
            dtype="float32",
            mode="w+",
            shape=(tot_num_windows, *self.signal_shape),
        )

        prep_mp = []

        for i, name in enumerate(self.preprocessors_name):
            path = os.path.join(self.dataset_folder, name + ".dat")
            prep_mp.append(
                np.memmap(
                    path,
                    dtype="float32",
                    mode="w+",
                    shape=(tot_num_windows, *self.preprocessors_shape[i]),
                )
            )

        ids, num_windows, start_index = [], [], []
        current_index = 0

        logger.info("Fetching the dataset ...")

        for subject_id, subject_records in enumerate(tqdm(self.get_subjects_records())):

            signal, labels = self.read_subject_record(subject_records)

            if signal is None and labels is None:
                logger.warning(
                    f"subject record - {subject_records} - is being discarded"
                )
                continue

            assert (
                signal.shape[0] == labels.shape[0]
            ), f"ERR: subject record - {subject_records} - signal and labels first dimension mismatch"
            assert (
                signal.ndim == 3
            ), f"ERR: subject record - {subject_records} - signal should be a 3D array of shape [ n_windows, n_channels, n_timestamps ]"

            labels_mp[current_index : current_index + signal.shape[0]] = labels.astype(
                np.int16
            )
            raw_mp[current_index : current_index + signal.shape[0]] = signal.astype(
                np.float32
            )

            raw_mp.flush()
            labels_mp.flush()

            for i, p_mp in enumerate(prep_mp):
                p_mp[current_index : current_index + signal.shape[0]] = (
                    self.preprocessors[i](signal).astype(np.float32)
                )

                p_mp.flush()

            ids.append(subject_id)
            num_windows.append(labels.shape[0])
            start_index.append(current_index)

            current_index += labels.shape[0]

        logger.info("Table creation ...")

        table = pd.DataFrame()

        table["subject_id"] = ids
        table["num_samples"] = num_windows
        table["start_index"] = start_index

        table = self.customize_table(table)

        table_path = os.path.join(self.dataset_folder, "table.csv")
        table.to_csv(table_path)

        self.table = table

        logger.info("Computing splitting parameters ...")

        split_path = os.path.join(self.dataset_folder, "splitting.pkl")
        train_subjects, valid_subjects, test_subjects = self.get_sets()

        with open(split_path, "wb") as f:
            pickle.dump(
                {
                    "train": train_subjects,
                    "valid": valid_subjects,
                    "test": test_subjects,
                },
                f,
            )

        logger.info("Computing scaling parameters ...")

        print(raw_mp.shape)
        raw_mean, raw_std = online_variance(raw_mp)

        print(raw_mean.shape, raw_std.shape)
        scaling_path = os.path.join(self.dataset_folder, "raw_scaling.npz")
        np.savez(scaling_path, mean=raw_mean, std=raw_std)

        for i, name in enumerate(self.preprocessors_name):
            scaling_path = os.path.join(self.dataset_folder, name + "_scaling.npz")
            p_mean, p_std = online_variance(prep_mp[i])

            print(p_mean.shape, p_std.shape)

            np.savez(scaling_path, mean=p_mean, std=p_std)

        # chmod 755 -R on the dataset directory
        chmod_recursive(
            self.dataset_folder,
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
        )


def online_variance(data):

    shape = data.shape[1:]

    n = 0
    mean = 0
    M2 = 0

    for x in tqdm(data):
        x = np.reshape(x, -1).astype(np.double)

        n = n + 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + delta * (x - mean)

    variance = M2 / (n - 1)

    variance = np.reshape(variance, shape)
    mean = np.reshape(mean, shape)

    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
