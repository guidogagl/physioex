import os 
from pathlib import Path
from os import listdir
from os.path import isfile, join

from dirhash import dirhash

from loguru import logger
import pkg_resources as pkg

import yaml

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm

import numpy as np
import pandas as pd
import h5py

from physioex.data.utils import read_cache, write_cache
from physioex.data.base import PhysioExDataset

from scipy.signal import butter, lfilter, resample
from sklearn.preprocessing import StandardScaler

home_directory = os.path.expanduser( '~' )
BASE_DIRECTORY = os.path.join( home_directory, 'dreem' )
BASE_DIRECTORY_H5 = os.path.join(BASE_DIRECTORY,  "h5" )

DATASET_HASH = "911138415522fa7ffe2d30ece62e3a12"

if not os.path.isdir(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

if not os.path.isdir(BASE_DIRECTORY_H5):
    os.mkdir(BASE_DIRECTORY_H5)

DODH_SETTINGS = {
    'h5_directory': os.path.join(BASE_DIRECTORY_H5, 'dodh' ),
}

DODO_SETTINGS = {
    'h5_directory': os.path.join(BASE_DIRECTORY_H5, 'dodo' ),
}

if not os.path.isdir(DODO_SETTINGS["h5_directory"]):
    os.mkdir(DODO_SETTINGS["h5_directory"])
if not os.path.isdir(DODH_SETTINGS["h5_directory"]):
    os.mkdir(DODH_SETTINGS["h5_directory"])


def download_dreem_dataset():
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    bucket_objects = client.list_objects(Bucket='dreem-dod-o')["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-O")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-o",
            Key=filename,
            Filename=DODO_SETTINGS["h5_directory"] + "/{}".format(filename)
        )
        
    bucket_objects = client.list_objects(Bucket='dreem-dod-h')["Contents"]
    print("\n Downloading H5 files and annotations from S3 for DOD-H")
    for bucket_object in tqdm.tqdm(bucket_objects):
        filename = bucket_object["Key"]
        client.download_file(
            Bucket="dreem-dod-h",
            Key=filename,
            Filename=DODH_SETTINGS["h5_directory"] + "/{}".format(filename)
        )

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class Dreem(PhysioExDataset):
    def __init__(self,  
                version : str = "dodh",
                use_cache : bool = True,
                preprocessors  = [
                    lambda data: np.multiply(data, 1e6), # Convert from V to uV
                    lambda data: butter_bandpass_filter(data, 0.3, 30, 250),
                    lambda data: resample(data, 100*30)
                    ],
                picks = ["C3_M2"]
                ):
        
        assert version in ["dodh", "dodo"], "version should be one of 'dodh'-'dodo'"
        
        self.version = version 
        self.window_dataset = None

        cache_path = "temp/dreem_" + version + ".pkl"
        Path("temp/").mkdir(parents=True, exist_ok=True)

        if use_cache:
            self.windows_dataset = read_cache( cache_path )
            
        if self.windows_dataset:
            return
            
        logger.info("Fetching the dataset..")
        
        try :
            found = (str(dirhash(BASE_DIRECTORY_H5, "md5", jobs= os.cpu_count())) == DATASET_HASH )
        except:
            found = False

        if not found:
            logger.info("Data not found, download dataset...")    
            download_dreem_dataset()
        
        data_path = DODO_SETTINGS["h5_directory"] if version == "dodo" else DODH_SETTINGS["h5_directory"] 

        files = [f for f in listdir(data_path)]

        # windowing
        X, y = [], []

        for f_name in files:
            if not f_name[-2:] == "h5":
                continue

            subj = h5py.File( os.path.join(data_path, f_name), "r")
            
            hyp = np.array(subj["hypnogram"][:], dtype=int)
            

            n_win = len(hyp)

            s_X, s_y = [], [] 

            for i in range(n_win):
                
                if hyp[i] == -1:
                    continue

                win_x = []
                for j, pick in enumerate(picks):
                    signal = subj["signals"]["eeg"][pick][ i*( 250*30 ): (i+1)*(250*30)]

                    for preprocess in preprocessors:
                        signal = preprocess(signal)
                    
                    win_x.append(signal)
                s_X.append( np.array( win_x ) )
                s_y.append( hyp[i] )

            y.append( np.array(s_y) )
            X.append( np.array(s_X) )

        windows_dataset = { "X": X, "y": y} 

        write_cache(cache_path, windows_dataset)
        
        self.windows_dataset = windows_dataset

    def split(self, fold : int = 0):
        
        window_dataset = [ self.windows_dataset["X"], self.windows_dataset["y"]]

        config = read_config()[ self.version ]["fold_%d" % fold]
        
        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []
        
        for indx in config["train"]:
            X_train.append( window_dataset[0][indx] )
            y_train.append( window_dataset[1][indx] )
        
        X_train = np.concatenate( X_train, axis = 0  )
        y_train = np.concatenate( y_train, axis = 0  )
        
        for indx in config["valid"]:
            X_valid.append( window_dataset[0][indx] )
            y_valid.append( window_dataset[1][indx] )
        
        X_valid = np.concatenate( X_valid, axis = 0  )
        y_valid = np.concatenate( y_valid, axis = 0  )
         
        for indx in config["test"]:
            X_test.append( window_dataset[0][indx] )
            y_test.append( window_dataset[1][indx] )

        X_test = np.concatenate( X_test , axis = 0 )
        y_test = np.concatenate( y_test , axis = 0 )
        
        logger.info("Train shape X " + str(X_train.shape) + ", y " + str( y_train.shape) )
        logger.info("Valid shape X " + str(X_valid.shape) + ", y " + str( y_valid.shape) )
        logger.info("Test shape X " + str(X_test.shape) + ", y " + str( y_test.shape) )

        train_set, valid_set, test_set = [],[],[]

        for i in range(len(y_train)):
            train_set.append( (X_train[i], y_train[i] ) )

        for i in range(len(y_valid)):
            valid_set.append( (X_valid[i], y_valid[i] ) )
        
        for i in range(len(y_test)):
            test_set.append( (X_test[i], y_test[i] ) )

        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set 

    def get_sets(self):
        return self.train_set, self.valid_set, self.test_set

@logger.catch
def read_config():
    config_file =  pkg.resource_filename(__name__, 'config/dreem.yaml')
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config 


if __name__ == "__main__":
    Dreem()