import os
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from physioex.data.constant import get_data_folder

input_dir = get_data_folder() + "/shhs/mat/"

output_dirs = {
    "raw": get_data_folder() + "/shhs/raw/",
    "xsleepnet": get_data_folder() + "/shhs/xsleepnet/",
}

csv_path = get_data_folder() + "/shhs/table.csv"

modalities = ["eeg", "eog", "emg"]


class Statistics:
    def __init__(self, num_modalities, raw_shape=[3000], preprocessed_shape=[29, 129]):

        self.current_raw_mean = np.zeros((num_modalities, *raw_shape))
        self.current_raw_std = np.zeros_like(self.current_raw_mean)

        self.current_preprocessed_mean = np.zeros((num_modalities, *preprocessed_shape))
        self.current_preprocessed_std = np.zeros_like(self.current_preprocessed_mean)

        self.count = np.zeros(num_modalities)

    def add_values(self, modality_index, num_samples, raw, preprocessed):

        raw_mean, raw_std = raw
        p_mean, p_std = preprocessed

        self.current_raw_mean[modality_index] += raw_mean
        self.current_raw_std[modality_index] += raw_std

        self.current_preprocessed_mean[modality_index] += p_mean
        self.current_preprocessed_std[modality_index] += p_std

        self.count[modality_index] += num_samples

    def get(self, modality_index):
        return (
            self.current_raw_mean[modality_index] / self.count[modality_index],
            np.sqrt(
                self.current_raw_std[modality_index] / self.count[modality_index]
                - np.square(
                    self.current_raw_mean[modality_index] / self.count[modality_index]
                )
            ),
            self.current_preprocessed_mean[modality_index] / self.count[modality_index],
            np.sqrt(
                self.current_preprocessed_std[modality_index]
                / self.count[modality_index]
                - np.square(
                    self.current_preprocessed_mean[modality_index]
                    / self.count[modality_index]
                )
            ),
        )


def process_file(filename, output_dir_raw, output_dir_preprocessed, statistics):
    subject_id, modality = (
        filename.split("_")[0][1:],
        filename.split("_")[1].split(".")[0],
    )

    try:
        with h5py.File(input_dir + filename, "r") as f:
            data = {key: f[key][()] for key in f.keys()}
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        exit(-1)

    # todo: check this correctly
    raw_data = np.transpose(data["X1"], (1, 0)).astype(np.float32)
    preprocessed_data = np.transpose(data["X2"], (2, 1, 0)).astype(np.float32)

    if modality == "eeg":
        y = np.reshape(data["label"], (-1)).astype(np.int16)
        fp = np.memmap(
            os.path.join(output_dir_raw, f"y_{int(subject_id)}.dat"),
            dtype="int16",
            mode="w+",
            shape=y.shape,
        )
        fp[:] = y[:]
        fp.flush()
        del fp

        fp = np.memmap(
            os.path.join(output_dir_preprocessed, f"y_{int(subject_id)}.dat"),
            dtype="int16",
            mode="w+",
            shape=y.shape,
        )
        fp[:] = y[:]
        fp.flush()
        del fp

    modality = modality.upper()

    new_filename = f"{modality}_{int( subject_id )}.dat"

    # Create memmap for raw data
    raw_memmap = np.memmap(
        os.path.join(output_dir_raw, new_filename),
        dtype="float32",
        mode="w+",
        shape=raw_data.shape,
    )
    raw_memmap[:] = raw_data[:]
    raw_memmap.flush()
    del raw_memmap

    # Create memmap for preprocessed data
    preprocessed_memmap = np.memmap(
        os.path.join(output_dir_preprocessed, new_filename),
        dtype="float32",
        mode="w+",
        shape=preprocessed_data.shape,
    )
    preprocessed_memmap[:] = preprocessed_data[:]
    preprocessed_memmap.flush()
    del preprocessed_memmap

    num_samples = raw_data.shape[0]

    raw_mean = np.sum(raw_data, axis=0)
    raw_std = np.sum(np.square(raw_data), axis=0)

    preprocessed_mean = np.sum(preprocessed_data, axis=0)
    preprocessed_std = np.sum(np.square(preprocessed_data), axis=0)

    return (
        subject_id,
        modality,
        num_samples,
        raw_mean,
        raw_std,
        preprocessed_mean,
        preprocessed_std,
    )


def process_files(input_dir, output_dir_raw, output_dir_preprocessed, csv_path):
    col_ids, col_samples, col_mod = [], [], []

    stats = Statistics(num_modalities=len(modalities))

    filenames = os.listdir(input_dir)

    results = Parallel(n_jobs=-1)(
        delayed(process_file)(filename, output_dir_raw, output_dir_preprocessed, stats)
        for filename in tqdm(filenames)
        if filename.endswith(".mat") and filename.startswith("n")
    )

    logger.info("Processing the jobs results")
    for result in results:
        if result is not None:
            (
                subject_id,
                modality,
                num_samples,
                raw_mean,
                raw_std,
                preprocessed_mean,
                preprocessed_std,
            ) = result

            raw = (raw_mean, raw_std)
            preprocessed = (preprocessed_mean, preprocessed_std)

            stats.add_values(
                modalities.index(modality.lower()), num_samples, raw, preprocessed
            )

            col_ids.append(subject_id)
            col_samples.append(num_samples)
            col_mod.append(modality)

    # Create a DataFrame and save to csv
    df = pd.DataFrame([])
    df["subject_id"] = col_ids
    df["modality"] = col_mod
    df["num_samples"] = col_samples

    # remove from the df all the modalities wich are not EEG
    df = df[df["modality"] == "EEG"]
    df = df.drop("modality", axis="columns")

    df.to_csv(csv_path, index=False)

    raw_mean, raw_std, prepr_mean, prepr_std = [], [], [], []
    for modality in modalities:
        rm, rstd, pm, pstd = stats.get(modalities.index(modality))

        raw_mean.append(np.array(rm).astype(np.float32))
        raw_std.append(np.array(rstd).astype(np.float32))

        prepr_mean.append(np.array(pm).astype(np.float32))
        prepr_std.append(np.array(pstd).astype(np.float32))

    np.savez(
        f"{output_dir_raw}/scaling.npz",
        mean=raw_mean,
        std=raw_std,
    )

    np.savez(
        f"{output_dir_preprocessed}/scaling.npz",
        mean=prepr_mean,
        std=prepr_std,
    )


if __name__ == "__main__":

    # check if the output_dirs exists and create them if not
    for key in output_dirs.keys():
        Path(output_dirs[key]).mkdir(parents=True, exist_ok=True)

    process_files(input_dir, output_dirs["raw"], output_dirs["xsleepnet"], csv_path)

    url = (
        "https://github.com/pquochuy/SleepTransformer/raw/main/shhs/data_split_eval.mat"
    )
    urlretrieve(url, get_data_folder() + "/shhs/data_split_eval.mat")
