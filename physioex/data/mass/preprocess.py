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

data_path = get_data_folder() + "/mass/"

input_dir = data_path + "mat/"

output_dirs = {
    "raw": data_path + "raw/",
    "xsleepnet": data_path + "xsleepnet/",
}

csv_path = data_path + "table.csv"

modalities = ["eeg", "eog", "emg"]


def process_file(filename, id_dictionary, output_dir_raw, output_dir_preprocessed):

    modality = filename.split("_")[2].split(".")[0]

    # get the subject id as the indx of the file in the filelist with same modality

    subject_id = id_dictionary[filename[:-8]]

    try:
        with h5py.File(input_dir + filename, "r") as f:
            data = {key: f[key][()] for key in f.keys()}
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        exit(-1)

    # todo: check this correctly
    raw_data = np.transpose(data["X1"], (1, 0)).astype(np.double)
    preprocessed_data = np.transpose(data["X2"], (2, 1, 0)).astype(np.double)

    # check if the data can be converted to float32, in case replace the data with the maximum value
    if (raw_data > np.finfo(np.float32).max).any():
        raw_data[raw_data > np.finfo(np.float32).max] = np.finfo(np.float32).max

    if (preprocessed_data > np.finfo(np.float32).max).any():
        preprocessed_data[preprocessed_data > np.finfo(np.float32).max] = np.finfo(
            np.float32
        ).max

    if (raw_data < np.finfo(np.float32).min).any():
        raw_data[raw_data < np.finfo(np.float32).min] = np.finfo(np.float32).min

    if (preprocessed_data < np.finfo(np.float32).min).any():
        preprocessed_data[preprocessed_data < np.finfo(np.float32).min] = np.finfo(
            np.float32
        ).min

    # convert the data to float32
    raw_data = raw_data.astype(np.float32)
    preprocessed_data = preprocessed_data.astype(np.float32)

    # assert no infinites are present in the data
    assert not np.isinf(
        raw_data
    ).any(), f"Raw data has infinities for subject {subject_id} and modality {modality}"
    assert not np.isinf(
        preprocessed_data
    ).any(), f"Preprocessed data has infinities for subject {subject_id} and modality {modality}"

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

    return (
        subject_id,
        modality,
        num_samples,
    )


def process_files(input_dir, output_dir_raw, output_dir_preprocessed, csv_path):
    col_ids, col_samples, col_mod = [], [], []

    filenames = os.listdir(input_dir)

    files_ids = [
        filename[:-8]
        for filename in filenames
        if filename.endswith(".mat") and filename.startswith("SS")
    ]
    files_ids = list(set(files_ids))

    file_to_id = {filename: idx for idx, filename in enumerate(files_ids)}

    results = Parallel(n_jobs = 1 )(
        delayed(process_file)(
            filename, file_to_id, output_dir_raw, output_dir_preprocessed
        )
        for filename in tqdm(filenames)
        if filename.endswith(".mat") and filename.startswith("SS")
    )

    logger.info("Processing the jobs results")
    for result in results:
        if result is not None:
            (
                subject_id,
                modality,
                num_samples,
            ) = result

            col_ids.append(subject_id)
            col_samples.append(num_samples)
            col_mod.append(modality)

    # Create a DataFrame and save to csv
    df = pd.DataFrame([])
    df["subject_id"] = col_ids
    df["modality"] = col_mod
    df["num_samples"] = col_samples

    df.to_csv(csv_path, index=False)


def read_raw_data(subject_id, modality, num_samples):
    raw_data = np.memmap(
        os.path.join(output_dirs["raw"], f"{modality.upper()}_{subject_id}.dat"),
        dtype="float32",
        mode="r",
        shape=(num_samples, 3000),
    )
    return raw_data


def read_xsleepnet_data(subject_id, modality, num_samples):
    xsleepnet_data = np.memmap(
        os.path.join(output_dirs["xsleepnet"], f"{modality.upper()}_{subject_id}.dat"),
        dtype="float32",
        mode="r",
        shape=(num_samples, 29, 129),
    )
    # return a copy of the data
    return xsleepnet_data


def compute_mean_std():
    table = pd.read_csv(csv_path)

    # sort table by subject_id and modality
    table = table.sort_values(by=["subject_id", "modality"])

    raw, xsleepnet = [], []

    for subject_id in tqdm(table["subject_id"].unique()):
        subject_df = table[table["subject_id"] == subject_id]
        num_samples = subject_df["num_samples"].values[0]
        modalities = subject_df["modality"].values

        mod_raw, mod_x = [], []
        for modality in modalities:

            mod_raw.append(read_raw_data(subject_id, modality, num_samples))
            mod_x.append(read_xsleepnet_data(subject_id, modality, num_samples))

        mod_raw = np.transpose(np.array(mod_raw), (1, 0, 2))
        mod_x = np.transpose(np.array(mod_x), (1, 0, 2, 3))

        raw.extend(mod_raw)
        xsleepnet.extend(mod_x)

    raw = np.array(raw).astype(np.double)

    raw_mean = np.mean(raw, axis=0).astype(np.float32)
    raw_std = np.std(raw, axis=0).astype(np.float32)

    # check that no nans are generated in the mean and std computation
    assert not np.isnan(raw_mean).any(), "Raw mean has nan values"
    assert not np.isnan(raw_std).any(), "Raw std has nan values"

    print(raw_mean.shape, raw_std.shape)

    np.savez(
        f"{output_dirs['raw']}/scaling.npz",
        mean=raw_mean,
        std=raw_std,
    )

    del raw

    xsleepnet = np.array(xsleepnet).astype(np.double)

    xsleepnet_mean = np.mean(xsleepnet, axis=0).astype(np.float32)
    xsleepnet_std = np.std(xsleepnet, axis=0).astype(np.float32)

    assert not np.isnan(xsleepnet_mean).any(), "Xsleepnet mean has nan values"
    assert not np.isnan(xsleepnet_std).any(), "Xsleepnet std has nan values"

    print(xsleepnet_mean.shape, xsleepnet_std.shape)

    np.savez(
        f"{output_dirs['xsleepnet']}/scaling.npz",
        mean=xsleepnet_mean,
        std=xsleepnet_std,
    )

    return



for key in output_dirs.keys():
    Path(output_dirs[key]).mkdir(parents=True, exist_ok=True)

logger.info("Processing files")
process_files(input_dir, output_dirs["raw"], output_dirs["xsleepnet"], csv_path)

logger.info("Computing mean and std")
compute_mean_std()

url = "https://github.com/pquochuy/xsleepnet/raw/master/mass/data_split_eval.mat"
urlretrieve(url, get_data_folder()+ "/mass/data_split_eval.mat")
