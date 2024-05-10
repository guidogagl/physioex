import os
from collections import Counter
from pathlib import Path
from typing import Callable, List

import numpy as np
import pkg_resources as pkg
import wfdb
import yaml
from dirhash import dirhash
from loguru import logger

from physioex.data.base import PhysioExDataset
from physioex.data.utils import read_cache, write_cache

from sklearn.model_selection import train_test_split


def download_mitdb(mitdb_dir):

    current_dir = os.getcwd()
    os.chdir(mitdb_dir)

    # Download the dataset into the current working dir using wfdb
    wfdb.dl_database("mitdb", os.getcwd())

    logger.info("Dataset downloaded.")
    os.chdir(current_dir)


def load_signal(mitdb_dir, picks, preprocessors, window_size: int = 2):

    config = read_config()
    label_dict = config["MITBIH_to_AAMI"]
    AAMI = config["AAMI_classes"]
    subjects = np.array(config["subjects"]).astype(int)

    records, annotations = [], []

    for subject in subjects:
        samp = wfdb.rdsamp(f"{mitdb_dir}/{subject}")
        annt = wfdb.rdann(f"{mitdb_dir}/{subject}", "atr")

        picked = []
        for pick in picks:

            signal = samp[0][:, samp[1]["sig_name"].index(pick)]

            # check if the preprocessors are not None before the for loop

            for preprocessor in preprocessors:
                signal = preprocessor(signal)

            picked.append(signal)

        picked = np.array(picked).reshape(len(picks), -1)

        # Segment the record into epochs
        # picked: n_pick, record_lenght
        #       -> record_lenght, n_pick
        #       -> n_windows, window_size, n_pick
        #       -> n_windows, n_pick, window_size

        picked = picked.T

        epoch_length = window_size * samp[1]["fs"]
        print(epoch_length)
        n_windows = picked.shape[0] // epoch_length
        picked = picked[: n_windows * epoch_length]

        picked = [picked[i : i + epoch_length] for i in range(n_windows)]
        picked = np.array(picked)

        picked = picked.reshape(n_windows, epoch_length, len(picks)).transpose(0, 2, 1)

        records.append(picked)

        epoch_labels = []
        for win in range(n_windows):
            # find the annotations in the epoch: annt.sample is the position of the annotation
            #                                   annt.symbol is the label of the annotation

            annt_indx = np.where(
                (annt.sample >= win * epoch_length)
                & (annt.sample < win * (epoch_length + 1))
            )[0]
            if epoch_anns := [annt.symbol[i] for i in annt_indx]:
                label = Counter(epoch_anns).most_common(1)[0][0]

                # if label is in MITBIH_classes convert it to AAMI_classes
                if label in label_dict.keys():
                    label = label_dict[label]
                    label = AAMI.index(label)
                else:
                    label = -1

                epoch_labels.append(label)

            else:
                epoch_labels.append(-1)
        epoch_labels = np.array(epoch_labels).astype(int)

        annotations.append(epoch_labels)

    # print the class distribution
    annt = np.concatenate(annotations)
    annt = annt[annt != -1]
    logger.info(f"Class distribution: {Counter(annt)}")

    return records, annotations


class MITBIH(PhysioExDataset):
    def __init__(
        self,
        version=None,
        use_cache: bool = True,
        picks: List[str] = ["MLII"],
        preprocessors: List[Callable] = [],
    ):

        for pick in picks:
            assert pick in ["MLII", "V5"], f"Pick {pick} not supported"

        self.windows_dataset = None

        self.train_set, self.valid_set, self.test_set = None, None, None

        cache_path = "temp/mitdb.pkl"
        Path("temp/").mkdir(parents=True, exist_ok=True)

        if use_cache:
            self.windows_dataset = read_cache(cache_path)

        if self.windows_dataset:
            return

        logger.info("Fetching the dataset..")
        home_dir = os.path.expanduser("~")
        mitdb_dir = os.path.join(home_dir, "mitdb")

        if not os.path.exists(mitdb_dir):
            os.makedirs(mitdb_dir)

        try:
            found = (
                str(dirhash(mitdb_dir, "md5", jobs=os.cpu_count()))
                == read_config()["HASH"]
            )
        except Exception:
            found = False

        if not found:
            logger.info("Data not found, download dataset...")

            download_mitdb(mitdb_dir)

        X, y = load_signal(mitdb_dir, picks, preprocessors)

        windows_dataset = {"X": X, "y": y}

        write_cache(cache_path, windows_dataset)

        self.windows_dataset = windows_dataset

    def split(self, fold: int = 0):

        window_dataset = [self.windows_dataset["X"], self.windows_dataset["y"]]

        config = read_config()
        subjects = config["subjects"]
        config = config["split"]

        X_train = np.concatenate(
            [window_dataset[0][subjects.index(indx)] for indx in config["train"]],
            axis=0,
        )
        y_train = np.concatenate(
            [window_dataset[1][subjects.index(indx)] for indx in config["train"]],
            axis=0,
        )

        groups = np.concatenate(
            [
                np.ones(window_dataset[0][subjects.index(indx)].shape[0])
                * subjects.index(indx)
                for indx in config["train"]
            ],
            axis=0,
        )

        X_test = np.concatenate(
            [window_dataset[0][subjects.index(indx)] for indx in config["test"]], axis=0
        )
        y_test = np.concatenate(
            [window_dataset[1][subjects.index(indx)] for indx in config["test"]], axis=0
        )

        # check the labels == -1 and remove those samples
        train_indx = np.where(y_train != -1)[0]
        X_train, y_train, groups = (
            X_train[train_indx],
            y_train[train_indx],
            groups[train_indx],
        )
        test_indx = np.where(y_test != -1)[0]
        X_test, y_test = X_test[test_indx], y_test[test_indx]

        # split the train set into train and valid stratifying on the labels
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.33, random_state=42 * fold, stratify=y_train
        )

        logger.info(f"Train shape X {str(X_train.shape)}, y {str(y_train.shape)}")
        logger.info(f"Valid shape X {str(X_valid.shape)}, y {str(y_valid.shape)}")
        logger.info(f"Test shape X {str(X_test.shape)}, y {str(y_test.shape)}")

        train_set, valid_set, test_set = [], [], []

        train_set.extend((X_train[i], y_train[i]) for i in range(len(y_train)))
        valid_set.extend((X_valid[i], y_valid[i]) for i in range(len(y_valid)))
        test_set.extend((X_test[i], y_test[i]) for i in range(len(y_test)))

        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set

    def get_sets(self):
        return self.train_set, self.valid_set, self.test_set


@logger.catch
def read_config():
    config_file = pkg.resource_filename(__name__, "config/mit-bih.yaml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config
