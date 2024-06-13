import os
import shutil
import stat
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pyedflib
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


def extract_large_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            extracted_path = zf.extract(member, extract_path)
            if member.create_system == 3:  # If the OS of creating system is Unix
                unix_attributes = member.external_attr >> 16
                if unix_attributes:
                    os.chmod(extracted_path, unix_attributes)
    os.remove(zip_path)


# Specifica la directory in cui desideri scaricare i file
dl_dir = get_data_folder()
dl_dir = os.path.join(dl_dir, "dcsm")
files = os.path.join(dl_dir, "data", "sleep", "DCSM")

# check if the dataset exists

if not os.path.exists(files):

    logger.info("Fetching the dataset...")
    os.makedirs(dl_dir, exist_ok=True)

    zip_file = dl_dir + "dcsm_dataset.zip"

    # URL del dataset
    if not os.path.exists(zip_file):
        download_file(
            "https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/dcsm_dataset.zip",
            zip_file,
        )

    # Estrai il file zip
    extract_large_zip(zip_file, dl_dir)

# chmod 755 -R
chmod_recursive(
    dl_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)
