import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import boto3
import h5py
import numpy as np
import pandas as pd

import pkg_resources as pkg
import tqdm
import yaml

from botocore import UNSIGNED
from botocore.client import Config

from loguru import logger
from scipy.signal import butter, lfilter, resample

from pathlib import Path
import os

from scipy import signal

home_directory = str( Path.home() )
BASE_DIRECTORY = os.path.join(home_directory, "dreem")
BASE_DIRECTORY_H5 = os.path.join(BASE_DIRECTORY, "h5")

DATASET_HASH = "911138415522fa7ffe2d30ece62e3a12"

if not os.path.isdir(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

if not os.path.isdir(BASE_DIRECTORY_H5):
    os.mkdir(BASE_DIRECTORY_H5)

DODH_SETTINGS = {
    "h5_directory": os.path.join(BASE_DIRECTORY_H5, "dodh"),
}

DODO_SETTINGS = {
    "h5_directory": os.path.join(BASE_DIRECTORY_H5, "dodo"),
}

if not os.path.isdir(DODO_SETTINGS["h5_directory"]):
    os.mkdir(DODO_SETTINGS["h5_directory"])
if not os.path.isdir(DODH_SETTINGS["h5_directory"]):
    os.mkdir(DODH_SETTINGS["h5_directory"])


def download_dreem_dataset():
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    bucket_objects = client.list_objects(Bucket="dreem-dod-o")["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-O")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-o",
            Key=filename,
            Filename=DODO_SETTINGS["h5_directory"] + "/{}".format(filename),
        )

    bucket_objects = client.list_objects(Bucket="dreem-dod-h")["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-H")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-h",
            Key=filename,
            Filename=DODH_SETTINGS["h5_directory"] + "/{}".format(filename),
        )


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


preprocessors=[
    lambda data: np.multiply(data, 1e6),  # Convert from V to uV
    lambda data: butter_bandpass_filter(data, 0.3, 30, 250),
    lambda data: resample(data, 100 * 30),
]


# xsleepnet preprocessing
def xsleepnet_preprocessing( sig ):
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
