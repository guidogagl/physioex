import os

import boto3
import numpy as np
import tqdm
from botocore import UNSIGNED
from botocore.client import Config
from loguru import logger
from scipy.signal import butter, lfilter, resample

DATASET_HASH = "911138415522fa7ffe2d30ece62e3a12"


def download_dreem_dataset(download_dir):

    dodo_dir = os.path.join(download_dir, "dodo")
    dodh_dir = os.path.join(download_dir, "dodh")

    os.makedirs(dodo_dir, exist_ok=True)
    os.makedirs(dodh_dir, exist_ok=True)

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    bucket_objects = client.list_objects(Bucket="dreem-dod-o")["Contents"]
    logger.info("Downloading H5 files and annotations from S3 for DOD-O")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-o",
            Key=filename,
            Filename=str(os.path.join(dodo_dir, filename)),
        )

    bucket_objects = client.list_objects(Bucket="dreem-dod-h")["Contents"]
    logger.info("Downloading H5 files and annotations from S3 for DOD-H")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-h",
            Key=filename,
            Filename=str(os.path.join(dodh_dir, filename)),
        )


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


preprocessors = [
    lambda data: np.multiply(data, 1e6),  # Convert from V to uV
    lambda data: butter_bandpass_filter(data, 0.3, 30, 250),
    lambda data: resample(data, 100 * 30),
]
