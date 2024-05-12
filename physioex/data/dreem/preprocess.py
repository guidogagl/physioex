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


def compute_splitting_parameters(path, subjects, shape):
    C3 = []
    EOG = []
    EMG = []

    for subject in tqdm(subjects):
        y = np.memmap(f"{path}/y_{subject}.dat", dtype="int16", mode="r")

        num_samples = y.shape[0]
        C3.extend(
            np.memmap(
                f"{path}/C3-M2_{subject}.dat",
                shape=(num_samples, *shape),
                dtype="float32",
                mode="r",
            )[:]
        )
        EOG.extend(
            np.memmap(
                f"{path}/EOG_{subject}.dat",
                shape=(num_samples, *shape),
                dtype="float32",
                mode="r",
            )[:]
        )
        EMG.extend(
            np.memmap(
                f"{path}/EMG_{subject}.dat",
                shape=(num_samples, *shape),
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
        f"{path}/scaling.npz",
        mean=[C3_mean, EOG_mean, EMG_mean],
        std=[C3_std, EOG_std, EMG_std],
    )


def create_table(version, num_samples):

    df = pd.DataFrame([])

    subjects_ids = np.arange(num_samples.shape[0])

    df["subject_id"] = subjects_ids
    df["num_samples"] = num_samples

    df.to_csv(str(Path.home()) + "/dreem/table_" + str(version) + ".csv")

    return


def process_data(data_path, version, output_path, preprocessing):

    output_dir = output_path + "/" + version + "/"

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]

    num_samples = []
    for subject_id, file in enumerate(files):

        subj = h5py.File(os.path.join(data_path, file), "r")
        hyp = np.array(subj["hypnogram"][:], dtype=int)

        n_win = len(hyp)

        X, y = [], []

        for i in range(n_win):

            if hyp[i] == -1:
                continue

            C3_M2 = subj["signals"]["eeg"]["C3_M2"][
                i * (250 * 30) : (i + 1) * (250 * 30)
            ]

            EOG = subj["signals"]["eog"]["EOG1"][i * (250 * 30) : (i + 1) * (250 * 30)]

            EMG = subj["signals"]["emg"]["EMG"][i * (250 * 30) : (i + 1) * (250 * 30)]

            for preprocess in utl.preprocessors:
                C3_M2 = preprocess(C3_M2)
                EOG = preprocess(EOG)
                EMG = preprocess(EMG)

            if preprocessing is not None:
                C3_M2 = preprocessing(C3_M2)
                EOG = preprocessing(EOG)
                EMG = preprocessing(EMG)

            X.append(np.array([C3_M2, EOG, EMG]))
            y.append(hyp[i])

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.int16).reshape(-1)

        C3, EOG, EMG = (
            X[:, 0],
            X[:, 1],
            X[:, 2],
        )

        # save the data into the output_dir

        fp = np.memmap(
            f"{output_dir}/C3-M2_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=C3.shape,
        )
        fp[:] = C3[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{output_dir}/EOG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EOG.shape,
        )
        fp[:] = EOG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{output_dir}/EMG_{subject_id}.dat",
            dtype="float32",
            mode="w+",
            shape=EMG.shape,
        )
        fp[:] = EMG[:]
        fp.flush()
        del fp

        fp = np.memmap(
            f"{output_dir}/y_{subject_id}.dat",
            dtype="int16",
            mode="w+",
            shape=y.shape,
        )
        fp[:] = y[:]
        fp.flush()
        del fp

        num_samples.append(y.shape[0])

    return np.array(num_samples).astype(int).reshape(-1)


def main():
    processed_paths = [
        str(Path.home() / "dreem" / "raw"),
        str(Path.home() / "dreem" / "xsleepnet"),
    ]
    versions = ["dodo", "dodh"]

    data_paths = [utl.DODO_SETTINGS["h5_directory"], utl.DODH_SETTINGS["h5_directory"]]

    for p in processed_paths:
        for v in versions:
            Path(p, v).mkdir(parents=True, exist_ok=True)

    logger.info("Fetching the dataset..")

    try:
        found = (
            str(dirhash(utl.BASE_DIRECTORY_H5, "md5", jobs=os.cpu_count()))
            == utl.DATASET_HASH
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        found = False

    if not found:
        logger.info("Data not found, download dataset...")
        utl.download_dreem_dataset()

    logger.info("Processing the data..")

    # raw data processing
    num_samples = {}
    num_samples["dodh"] = process_data(
        data_paths[1], "dodh", processed_paths[0], preprocessing=None
    )
    num_samples["dodo"] = process_data(
        data_paths[0], "dodo", processed_paths[0], preprocessing=None
    )

    # xsleepnet data processing
    process_data(
        data_paths[1],
        "dodh",
        processed_paths[1],
        preprocessing=utl.xsleepnet_preprocessing,
    )
    process_data(
        data_paths[0],
        "dodo",
        processed_paths[1],
        preprocessing=utl.xsleepnet_preprocessing,
    )

    logger.info("Creating the tables..")

    create_table("dodh", num_samples["dodh"])
    create_table("dodo", num_samples["dodo"])

    logger.info("Computing the splitting parameters")

    # raw data scaling
    compute_splitting_parameters(
        processed_paths[0] + "/dodh/",
        subjects=np.arange(len(num_samples["dodh"])).astype(int),
        shape=[3000],
    )
    compute_splitting_parameters(
        processed_paths[0] + "/dodo/",
        subjects=np.arange(len(num_samples["dodo"])).astype(int),
        shape=[3000],
    )

    # xsleepnet data scaling
    compute_splitting_parameters(
        processed_paths[1] + "/dodh/",
        subjects=np.arange(len(num_samples["dodh"])).astype(int),
        shape=[29, 129],
    )
    compute_splitting_parameters(
        processed_paths[1] + "/dodo/",
        subjects=np.arange(len(num_samples["dodo"])).astype(int),
        shape=[29, 129],
    )


if __name__ == "__main__":
    main()
