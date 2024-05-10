import physioex.data.dreem.utils as utl
from physioex.data.utils import read_config

from loguru import logger
import os
from dirhash import dirhash

import pandas as pd

import numpy as np
import h5py

from pathlib import Path
from tqdm import tqdm

picks = ["C3_M2", "EOG", "EMG"]

# ---- Fetching the dataset -------

processed_path = [
    str(Path.home()) + "/dreem/raw/",
    str(Path.home()) + "/dreem/xsleepnet/",
]
versions = ["dodo", "dodh"]

for p in processed_path:
    for v in versions:
        # check if the path exists and in case create it
        Path(p + "/" + v + "/").mkdir(parents=True, exist_ok=True)

data_paths = [utl.DODO_SETTINGS["h5_directory"], utl.DODH_SETTINGS["h5_directory"]]

logger.info("Fetching the dataset..")

try:
    found = (
        str(dirhash(utl.BASE_DIRECTORY_H5, "md5", jobs=os.cpu_count()))
        == utl.DATASET_HASH
    )
except:
    found = False

if not found:
    logger.info("Data not found, download dataset...")
    utl.download_dreem_dataset()

for data_path, version in zip(data_paths, versions):

    files = [f for f in os.listdir(data_path)]

    subject_id = 0

    for f_name in tqdm(files):

        if not f_name[-2:] == "h5":
            continue

        subj = h5py.File(os.path.join(data_path, f_name), "r")
        hyp = np.array(subj["hypnogram"][:], dtype=int)

        n_win = len(hyp)

        X, y = [], []

        for i in range(n_win):

            if hyp[i] == -1:
                continue

            win_x = []

            for j, pick in enumerate(picks):

                C3_M2 = subj["signals"]["eeg"]["C3_M2"][
                    i * (250 * 30) : (i + 1) * (250 * 30)
                ]

                EOG = subj["signals"]["eog"]["EOG1"][
                    i * (250 * 30) : (i + 1) * (250 * 30)
                ]

                EMG = subj["signals"]["emg"]["EMG"][
                    i * (250 * 30) : (i + 1) * (250 * 30)
                ]

                for preprocess in utl.preprocessors:
                    C3_M2 = preprocess(C3_M2)
                    EOG = preprocess(EOG)
                    EMG = preprocess(EMG)

                win_x.append([C3_M2, EOG, EMG])

            X.append(np.array(win_x))
            y.append(hyp[i])

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.int16)

        C3, EOG, EMG = (
            X[:, 0],
            X[:, 1],
            X[:, 2],
        )

        # raw
        fp = np.memmap(
            f"{processed_path[0]}/{version}/C3-M2_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=C3.shape,
        )
        fp[:] = C3[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[0]}/{version}/EOG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EOG.shape,
        )
        fp[:] = EOG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[0]}/{version}/EMG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EMG.shape,
        )
        fp[:] = EMG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[0]}/{version}/y_{subject_id}.dat",
            dtype="int16",
            mode="w+",
            shape=y.shape,
        )
        fp[:] = y[:]
        fp.flush()
        del fp

        # perform the xsleepnet preprocesing and save the correspondig subject files

        X_tf = []
        for i in range(X.shape[0]):
            x = []
            for j in range(X.shape[1]):
                x.append(utl.xsleepnet_preprocessing(X[i, j]))
            X_tf.append(x)

        X = np.array(X_tf).astype(np.float32)

        C3, EOG, EMG = X[:, 0], X[:, 1], X[:, 2]

        # raw
        fp = np.memmap(
            f"{processed_path[1]}/{version}/C3-M2_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=C3.shape,
        )
        fp[:] = C3[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[1]}/{version}/EOG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EOG.shape,
        )
        fp[:] = EOG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[1]}/{version}/EMG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EMG.shape,
        )
        fp[:] = EMG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{processed_path[1]}/{version}/y_{subject_id}.dat",
            dtype="int16",
            mode="w+",
            shape=y.shape,
        )
        fp[:] = y[:]
        fp.flush()
        del fp

        subject_id = subject_id + 1

# ----- create the tables for each dataset version ----

logger.info("Creating the tables..")

for vers_indx, v in enumerate(versions):

    files = [f for f in os.listdir(data_paths[vers_indx])]
    num_subjects = 0
    for file in files:
        if file[-2:] == "h5":
            num_subjects += 1

    # get the subject list
    subjects = np.arange(num_subjects).astype(int)

    # read the subject y file to get the number of samples of that subject
    num_samples = []
    for i in subjects:
        y = np.memmap(f"{processed_path[0]}/{v}/y_{i}.dat", dtype="int16", mode="r")

        num_samples.append(y.shape[0])

    table = pd.DataFrame([])

    table["subject_id"] = subjects
    table["num_samples"] = np.array(num_samples).astype(int)

    table.to_csv(str(Path.home()) + "/dreem/table_" + str(v) + ".csv")


# ------ compute the splitting parameters for each version ----

logger.info("Computing the splitting parameters")

config = read_config("config/dreem.yaml")

for vers_indx, v in enumerate(versions):

    logger.info(f"Processing {v} dataset")

    files = [f for f in os.listdir(data_paths[vers_indx])]
    num_subjects = 0
    for file in files:
        if file[-2:] == "h5":
            num_subjects += 1

    # get the subject list
    subjects = np.arange(num_subjects).astype(int)

    C3 = []
    EOG = []
    EMG = []

    for subject in tqdm(subjects):
        y = np.memmap(
            f"{processed_path[0]}/{v}/y_{subject}.dat", dtype="int16", mode="r"
        )

        num_samples = y.shape[0]

        C3.extend(
            np.memmap(
                f"{processed_path[0]}/{v}/C3-M2_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )
        EOG.extend(
            np.memmap(
                f"{processed_path[0]}/{v}/EOG_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )
        EMG.extend(
            np.memmap(
                f"{processed_path[0]}/{v}/EMG_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )

    C3, EOG, EMG = (
        np.array(C3).astype(np.float32),
        np.array(EOG).astype(np.float32),
        np.array(EMG).astype(np.float32),
    )

    C3_mean, C3_std = np.mean(C3, axis=0), np.std(C3, axis=0)
    EOG_mean, EOG_std = np.mean(EOG, axis=0), np.std(EOG, axis=0)
    EMG_mean, EMG_std = np.mean(EMG, axis=0), np.std(EMG, axis=0)

    np.savez(
        f"{processed_path[0]}/{v}/scaling.npz",
        mean=[C3_mean, EOG_mean, EMG_mean],
        std=[C3_std, EOG_std, EMG_std],
    )

    C3 = []
    EOG = []
    EMG = []

    for subject in tqdm(subjects):
        y = np.memmap(
            f"{processed_path[0]}/{v}/y_{subject}.dat", dtype="int16", mode="r"
        )

        num_samples = y.shape[0]
        C3.extend(
            np.memmap(
                f"{processed_path[1]}/{v}/C3-M2_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )
        EOG.extend(
            np.memmap(
                f"{processed_path[1]}/{v}/EOG_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )
        EMG.extend(
            np.memmap(
                f"{processed_path[1]}/{v}/EMG_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )

    C3, EOG, EMG = (
        np.array(C3).astype(np.float32),
        np.array(EOG).astype(np.float32),
        np.array(EMG).astype(np.float32),
    )

    C3_mean, C3_std = np.mean(C3, axis=0), np.std(C3, axis=0)
    EOG_mean, EOG_std = np.mean(EOG, axis=0), np.std(EOG, axis=0)
    EMG_mean, EMG_std = np.mean(EMG, axis=0), np.std(EMG, axis=0)

    np.savez(
        f"{processed_path[1]}/{v}/scaling.npz",
        mean=[C3_mean, EOG_mean, EMG_mean],
        std=[C3_std, EOG_std, EMG_std],
    )
