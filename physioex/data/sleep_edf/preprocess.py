from braindecode.datasets import SleepPhysionet as SP
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

import numpy as np
import h5py
from scipy import signal

from tqdm import tqdm
import pandas as pd

from physioex.data.utils import read_config

from tqdm import tqdm

# read the file data/sleep-edf-split.mat
from scipy.io import loadmat

from urllib.request import urlretrieve
from pathlib import Path


from loguru import logger


# xsleepnet preprocessing
def xsleepnet_preprocessing(sig):
    # transform each signal into its spectrogram ( fast )
    # nfft 256, noverlap 1, win 2, fs 100, hamming window
    _, _, Sxx = signal.spectrogram(
        sig.reshape(-1),
        fs=100,
        window="hamming",
        nperseg=200,
        noverlap=100,
        nfft=256,
    )

    # log_10 scale the spectrogram safely (using epsilon)
    Sxx = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

    Sxx = np.transpose(Sxx, (1, 0))

    return Sxx


#### ----------- config loading & constants ------------ ####

logger.info("Loading config file")
config = read_config("config/sleep-edf.yaml")

preprocessors = [
    Preprocessor(lambda data: np.multiply(data, 1e6), apply_on_array=True),
    Preprocessor("filter", l_freq=0.3, h_freq=40),
]

home_dir = str(Path.home())

data_folder = home_dir + "/mne_data/physionet-sleep-data/"

logger.info("Creating the folder for the data")

# create the folder for the processed data if not exists
Path(data_folder).mkdir(parents=True, exist_ok=True)

xsleepnet_folder = home_dir + "/mne_data/sleep-edf-xsleepnet/"
Path(xsleepnet_folder).mkdir(parents=True, exist_ok=True)

raw_folder = home_dir + "/mne_data/sleep-edf-raw/"
# create the folder for the raw data if not exists
Path(raw_folder).mkdir(parents=True, exist_ok=True)


# -------------- dataset creation ----------------- #

logger.info("Fetching the dataset")

dataset = SP(
    subject_ids=config["subjects_v2018"],
    recording_ids=[1, 2],
    crop_wake_mins=30,
    load_eeg_only=False,
)

# filtering
preprocess(dataset, preprocessors, n_jobs=-1)

# windowing
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=30 * 100,
    window_stride_samples=30 * 100,
    preload=True,
    mapping=config["mapping"],
    picks=["Fpz-Cz", "EOG horizontal", "EMG submental"],
    n_jobs=-1,
)

windows_dataset = windows_dataset.split("subject")


# available prepocessing options:
# - raw
# - xsleepnet

# loop over the subjects


for key in tqdm(windows_dataset.keys()):
    subject_dataset = windows_dataset[key]

    # raw preprocessing
    EEG = []
    EOG = []
    EMG = []

    # xsleepnet preprocessing
    EEG_tf = []
    EOG_tf = []
    EMG_tf = []

    y = []

    # loop over the windows of subjects
    for i in range(len(subject_dataset)):
        sig, label, _ = subject_dataset[i]

        y.append(label)

        sig, eog, emg = sig

        # save raw window data
        EEG.append(sig)
        EOG.append(eog)
        EMG.append(emg)

        # save xsleepnet processed window data
        EEG_tf.append(xsleepnet_preprocessing(sig))
        EOG_tf.append(xsleepnet_preprocessing(eog))
        EMG_tf.append(xsleepnet_preprocessing(emg))

    EEG = np.array(EEG).astype(np.float32)
    EOG = np.array(EOG).astype(np.float32)
    EMG = np.array(EMG).astype(np.float32)

    EEG_tf = np.array(EEG_tf).astype(np.float32)
    EOG_tf = np.array(EOG_tf).astype(np.float32)
    EMG_tf = np.array(EMG_tf).astype(np.float32)

    y = np.array(y).astype(np.int16)

    # save the data for each preprocessing using numpy memmap

    # raw
    fp = np.memmap(
        f"{raw_folder}/Fpz-Cz_{key}.dat", dtype="float32", mode="w+", shape=EEG.shape
    )
    fp[:] = EEG[:]
    fp.flush()
    del fp

    fp = np.memmap(
        f"{raw_folder}/EOG_{key}.dat", dtype="float32", mode="w+", shape=EOG.shape
    )
    fp[:] = EOG[:]
    fp.flush()
    del fp

    fp = np.memmap(
        f"{raw_folder}/EMG_{key}.dat", dtype="float32", mode="w+", shape=EMG.shape
    )
    fp[:] = EMG[:]
    fp.flush()
    del fp

    # xsleepnet
    fp = np.memmap(
        f"{xsleepnet_folder}/Fpz-Cz_{key}.dat",
        dtype="float32",
        mode="w+",
        shape=EEG_tf.shape,
    )
    fp[:] = EEG_tf[:]
    fp.flush()
    del fp

    fp = np.memmap(
        f"{xsleepnet_folder}/EOG_{key}.dat",
        dtype="float32",
        mode="w+",
        shape=EOG_tf.shape,
    )
    fp[:] = EOG_tf[:]
    fp.flush()
    del fp

    fp = np.memmap(
        f"{xsleepnet_folder}/EMG_{key}.dat",
        dtype="float32",
        mode="w+",
        shape=EMG_tf.shape,
    )
    fp[:] = EMG_tf[:]
    fp.flush()
    del fp

    # y
    fp = np.memmap(f"{raw_folder}/y_{key}.dat", dtype="int16", mode="w+", shape=y.shape)
    fp[:] = y[:]
    fp.flush()
    del fp

    fp = np.memmap(
        f"{xsleepnet_folder}/y_{key}.dat", dtype="int16", mode="w+", shape=y.shape
    )
    fp[:] = y[:]
    fp.flush()
    del fp

# -------------- create the table for the dataset -------------- #


logger.info("Creating the table for the dataset")
# download the xls file
urlretrieve(config["xls_url"], data_folder + "../sleep-edf-subjects.xls")
# read it
df = pd.read_excel(data_folder + "../sleep-edf-subjects.xls")

table = []

for i in np.unique(df["subject"].values):
    row = [
        i,
        df[df["subject"] == i]["age"].values.astype(int)[0],
    ]

    sex = df[df["subject"] == i]["sex (F=1)"].astype(int).values[0]
    sex = sex if sex == 1 else 0

    row.append(sex)

    # read the y file to get the number of samples
    y = np.memmap(f"{raw_folder}/y_{i}.dat", dtype="int16", mode="r")

    num_samples = y.shape[0]

    row.append(num_samples)

    table.append(row)

table = pd.DataFrame(table, columns=["subject_id", "age", "sex", "num_samples"])
table.to_csv(data_folder + "../sleep-edf-table.csv", index=False)

# -------------- download the splits for the edf-78 and edf-20 versions of the dataset  -------------- #

logger.info("Downlaoding the splits for the dataset version 2018")

urlretrieve(config["splits_url"], data_folder + "../sleep-edf-split.mat")
data = loadmat(data_folder + "../sleep-edf-split.mat")

logger.info("Downloading the splits for the dataset version 2013")

urlretrieve(config["splits_2013_url"], data_folder + "../sleep-edf-split-2013.mat")
data = loadmat(data_folder + "../sleep-edf-split-2013.mat")

# ------------- use the downloaded splits to save the standardization parameters for the datasets ------------ #

for version in ["2018", "2013"]:

    logger.info("Creating the splits for the dataset version " + version)

    split_matrix = loadmat(data_folder + "../sleep-edf-split.mat")
    subjects = list(config["subjects_v" + version])

    EEG = []
    EOG = []
    EMG = []

    EEG_tf = []
    EOG_tf = []
    EMG_tf = []

    for subject in tqdm(subjects):
        y = np.memmap(f"{raw_folder}/y_{subject}.dat", dtype="int16", mode="r")
        num_samples = y.shape[0]

        EEG.extend(
            np.memmap(
                f"{raw_folder}/Fpz-Cz_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )
        EOG.extend(
            np.memmap(
                f"{raw_folder}/EOG_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )
        EMG.extend(
            np.memmap(
                f"{raw_folder}/EMG_{subject}.dat",
                shape=(num_samples, 3000),
                dtype="float32",
                mode="r",
            )[:]
        )

    EEG, EOG, EMG = (
        np.array(EEG).astype(np.float32),
        np.array(EOG).astype(np.float32),
        np.array(EMG).astype(np.float32),
    )

    EEG_mean, EEG_std = np.mean(EEG, axis=0), np.std(EEG, axis=0)
    EOG_mean, EOG_std = np.mean(EOG, axis=0), np.std(EOG, axis=0)
    EMG_mean, EMG_std = np.mean(EMG, axis=0), np.std(EMG, axis=0)

    np.savez(
        f"{raw_folder}/scaling_{version}.npz",
        mean=[EEG_mean, EOG_mean, EMG_mean],
        std=[EEG_std, EOG_std, EMG_std],
    )

    del EEG, EOG, EMG

    for subject in tqdm(subjects):
        y = np.memmap(f"{xsleepnet_folder}/y_{subject}.dat", dtype="int16", mode="r")
        num_samples = y.shape[0]

        EEG_tf.extend(
            np.memmap(
                f"{xsleepnet_folder}/Fpz-Cz_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )
        EOG_tf.extend(
            np.memmap(
                f"{xsleepnet_folder}/EOG_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )
        EMG_tf.extend(
            np.memmap(
                f"{xsleepnet_folder}/EMG_{subject}.dat",
                shape=(num_samples, 29, 129),
                dtype="float32",
                mode="r",
            )[:]
        )

    EEG_tf, EOG_tf, EMG_tf = (
        np.array(EEG_tf).astype(np.float32),
        np.array(EOG_tf).astype(np.float32),
        np.array(EMG_tf).astype(np.float32),
    )

    EEG_tf_mean, EEG_tf_std = np.mean(EEG_tf, axis=0), np.std(EEG_tf, axis=0)
    EOG_tf_mean, EOG_tf_std = np.mean(EOG_tf, axis=0), np.std(EOG_tf, axis=0)
    EMG_tf_mean, EMG_tf_std = np.mean(EMG_tf, axis=0), np.std(EMG_tf, axis=0)

    # save the mean and std for each signal
    np.savez(
        f"{xsleepnet_folder}/scaling_{version}.npz",
        mean=[EEG_tf_mean, EOG_tf_mean, EMG_tf_mean],
        std=[EEG_tf_std, EOG_tf_std, EMG_tf_std],
    )

    del EEG_tf, EOG_tf, EMG_tf
