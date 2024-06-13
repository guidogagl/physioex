import os
import stat
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pyedflib
import rarfile
import requests
from loguru import logger
from scipy.signal import butter, filtfilt, resample, spectrogram
from tqdm import tqdm

from physioex.data.constant import get_data_folder


def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def chmod_recursive(path, mode):
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, mode)
        for filename in filenames:
            os.chmod(os.path.join(dirpath, filename), mode)


fs = 200


def get_labels(filepath):
    # open the txt file to get the labels
    with open(filepath + "_1.txt", "r") as f:
        labels = f.readlines()

    labels = np.array(
        [int(label.strip()) for label in labels if label.strip().isdigit()]
    ).astype(int)
    return labels


@logger.catch
def read_edf(file_path):

    labels = get_labels(file_path)

    f = pyedflib.EdfReader(file_path + ".rec")

    buffer = []

    try:
        i = f.getSignalLabels().index("C3-A2")
    except:
        try:
            i = f.getSignalLabels().index("C3-M2")
        except:
            logger.warning(
                f"No valid channels found for {filepath} available are {f.getSignalLabels()}"
            )
            return None, None

    signal = f.readSignal(i)

    f._close()

    # windowing of the signal
    signal = signal.reshape(-1, fs)
    num_windows = signal.shape[0] // 30
    signal = signal[: num_windows * 30]
    signal = signal.reshape(-1, 30 * fs)

    signal = resample(signal, num=30 * 100, axis=1)
    buffer.append(signal)

    buffer = np.array(buffer)
    n_samples = min(labels.shape[0], buffer.shape[1])

    buffer, labels = buffer[:, :n_samples, :], labels[:n_samples]

    mask = np.logical_and(labels != 6, labels != 7)
    buffer, labels = buffer[:, mask], labels[mask]

    # map the labels to the new values
    labels = np.array(
        list(
            map(
                lambda x: (
                    0
                    if x == 0
                    else 4 if x == 1 else 1 if x == 2 else 2 if x == 3 else 3
                ),
                labels,
            )
        )
    )

    # print( f"Buffer shape {buffer.shape} labels shape {labels.shape}" )
    buffer = np.transpose(buffer, (1, 0, 2))
    return buffer, labels


# xsleepnet preprocessing
def xsleepnet_preprocessing(sig):
    # transform each signal into its spectrogram ( fast )
    # nfft 256, noverlap 1, win 2, fs 100, hamming window
    _, _, Sxx = spectrogram(
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


def save_memmaps(path, signal, labels, subject_id):

    for i, modality in enumerate(["EEG"]):
        file_path = os.path.join(path, f"{modality}_{subject_id}.dat")
        fp = np.memmap(file_path, dtype="float32", mode="w+", shape=signal[i].shape)
        fp[:] = signal[i][:]
        fp.flush()
        del fp

    labels_path = os.path.join(path, f"y_{subject_id}.dat")
    fp = np.memmap(labels_path, dtype="int16", mode="w+", shape=labels.shape)
    fp[:] = labels[:]
    fp.flush()
    del fp
    return


subgroups = [list(range(1, 101)), list(range(1, 9)), list(range(1, 11))]

discarded_subjects = []

# Specifica la directory in cui desideri scaricare i file
dl_dir = get_data_folder()
dl_dir += "isruc/"

files = [dl_dir + f"subgroup{i}/" for i in range(1, 4)]

i = 0
# check if the dataset exists
for file, subgroup in zip(files, subgroups):

    if i == 0:
        id = "I"
    elif i == 1:
        id = "II"
    else:
        id = "III"

    subgroup_id = f"subgroup{id}"

    i += 1

    os.makedirs(file, exist_ok=True)

    # URL del dataset
    for subject in subgroup:

        url = f"http://dataset.isr.uc.pt/ISRUC_Sleep/{subgroup_id}/{subject}.rar"
        zip_file = f"{file}/{subject}.rar"

        # check if the dataset is already there

        if not os.path.exists(f"{file}/{subject}/"):
            # Scarica il dataset
            download_file(url, zip_file)

            with rarfile.RarFile(zip_file, "r") as rar_ref:
                rar_ref.extractall(file)

            # Rimuovi il file rar
            os.remove(zip_file)

# chmod 755 -R
chmod_recursive(
    dl_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)

#
logger.info("Processing the data...")

os.makedirs(dl_dir + "raw", exist_ok=True)
os.makedirs(dl_dir + "xsleepnet", exist_ok=True)

raw_data, xsleepnet_data = [], []

subject_id = 0

for subgroup_id in [0, 2]:

    group_subjects = subgroups[subgroup_id]

    for subject in tqdm(group_subjects):

        filepath = f"{dl_dir}/subgroup{ subgroup_id + 1}/{subject}/{subject}"
        signal, labels = read_edf(filepath)

        if signal is None:
            discarded_subjects.append(subject_id)
            logger.warning(f"Subject {subject_id} discarded")
            subject_id += 1

            continue

        xsleepnet = np.zeros((signal.shape[0], 1, 29, 129))
        for i in range(len(signal)):
            for m in range(1):
                xsleepnet[i, m] = xsleepnet_preprocessing(signal[i, m])

        save_memmaps(dl_dir + "raw", signal, labels, subject_id)
        save_memmaps(dl_dir + "xsleepnet", xsleepnet, labels, subject_id)

        xsleepnet_data.extend(xsleepnet)
        raw_data.extend(signal)
        subject_id += 1


logger.info("Table creation...")

sub1_desc = pd.read_excel(
    "http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_I_Submission.xlsx",
    header=2,
)
sub3_desc = pd.read_excel(
    "http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_III_Submission.xlsx",
    header=2,
)

sub1_desc["Subject"] = sub1_desc["Subject"] - 1
sub3_desc["Subject"] = sub3_desc["Subject"] - 1 + sub1_desc.shape[0]

frames = [sub1_desc, sub3_desc]
desc = pd.concat(frames)

table = pd.DataFrame([])

table["subject_id"] = desc["Subject"]
table["age"] = desc["Age"]
table["gender"] = desc["Sex"]
table["num_samples"] = desc["Epoches"]

# discard all the subject in discarded_subject in the table

table = table[~table["subject_id"].isin(discarded_subjects)]

table.to_csv(dl_dir + "/table.csv")

logger.info("Data processing completed, computing standardization")

raw_data = np.array(raw_data)
xsleepnet_data = np.array(xsleepnet_data)

print(xsleepnet_data.shape, raw_data.shape)

raw_mean, raw_std = np.mean(raw_data, axis=0), np.std(raw_data, axis=0)
xsleepnet_mean, xsleepnet_std = np.mean(xsleepnet_data, axis=0), np.std(
    xsleepnet_data, axis=0
)

logger.info("Saving scaling parameters")
# save the mean and std for each signal
np.savez(
    f"{dl_dir}/raw/scaling.npz",
    mean=raw_mean,
    std=raw_std,
)

np.savez(
    f"{dl_dir}/xsleepnet/scaling.npz",
    mean=xsleepnet_mean,
    std=xsleepnet_std,
)

print(raw_mean.shape, raw_std.shape, xsleepnet_mean.shape, xsleepnet_std.shape)

logger.info("Saving splitting parameters")
# computing the splitting subjects train valid test with ratio 0.7 0.15 0.15
# use a setted seed for reproducibility

np.random.seed(42)

train_subjects = np.random.choice(
    table["subject_id"], size=int(table.shape[0] * 0.7), replace=False
)
valid_subjects = np.setdiff1d(table["subject_id"], train_subjects, assume_unique=True)
test_subjects = np.random.choice(
    valid_subjects, size=int(table.shape[0] * 0.15), replace=False
)
valid_subjects = np.setdiff1d(valid_subjects, test_subjects, assume_unique=True)

print(train_subjects.shape, valid_subjects.shape, test_subjects.shape)

np.savez(
    f"{dl_dir}/splitting.npz",
    train=train_subjects,
    valid=valid_subjects,
    test=test_subjects,
)
