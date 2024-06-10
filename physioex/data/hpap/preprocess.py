import os
import subprocess

from loguru import logger

import pandas as pd 
import numpy as np

import pyedflib

from scipy.signal import resample, spectrogram, butter, filtfilt

from tqdm import tqdm 
from physioex.data.constant import get_data_folder

#0 - Wake --> map 0
#1 - REM ---> map 4
#2 - Stage 1 ---> map 1
#3 - Stage 2 ---> map 2
#4 - Stage 3 ---> map 3
#5 - Stage 4 ---> map 3

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def read_edf(file_path):
    
    # open the txt file to get the labels
    with open( file_path + "_stage.txt", "r") as f:
        labels = f.readlines()
    
    # remove \n from the labels
    labels = list( map( lambda x: int(x.strip()), labels ) )
    labels = np.array( labels )
    
    n_samples = labels.shape[0]
    
    f = pyedflib.EdfReader(file_path + ".rec")
    
    buffer = []
    
    
    for indx, modality in enumerate( ["C3A2", "EOG", "EMG", "ECG"]) :
        if modality == "EOG":
            
            left = f.getSignalLabels().index( "Lefteye" )
            right = f.getSignalLabels().index( "RightEye" )
            
            if f.getSampleFrequency(left) != f.getSampleFrequency(right):
                logger.error("Sampling frequency of EOG signals is different")
                exit()
            
            fs = int(f.getSampleFrequency(left))
            
            signal = (f.readSignal( left ) - f.readSignal( right )).reshape( -1, fs)
            num_windows = signal.shape[0] // 30 
            
            #print( f"Modality {modality} fs {fs} num windows {num_windows} shape {signal.shape}" )
            
            signal = signal[:num_windows * 30]
            signal = signal.reshape( -1, 30 * fs)

        else:
            i = f.getSignalLabels().index( modality )
            fs = int( f.getSampleFrequency(i) )
            signal = f.readSignal( i ).reshape(-1, fs)
            
            #print( f"Modality {modality} fs {fs} shape {signal.shape}" )
            # consider windows of 30 seconds, discard the last epoch if not fit
            num_windows = signal.shape[0] // 30 
            
            #print( f"Modality {modality} fs {fs} num windows {num_windows} shape {signal.shape}" )

            signal = signal[:num_windows * 30]
            signal = signal.reshape( -1, 30 * fs)
        
        
        # resample the signal at 100Hz
        signal = resample( signal, num = 30 * 100, axis = 1)
        # pass band the signal between 0.3 and 40 Hz        
        signal = bandpass_filter( signal, 0.3, 40, 100)
        
        buffer.append( signal )
    f._close()
    
    
    buffer = np.array( buffer )
    n_samples = min( n_samples, buffer.shape[1] )
    
    buffer, labels = buffer[:, :n_samples, :], labels[:n_samples]

    
    mask = np.logical_and( labels != 6, labels != 7 )
    buffer, labels = buffer[:, mask], labels[mask]
    
    # map the labels to the new values
    labels = np.array( list( map( lambda x: 0 if x == 0 else 4 if x == 1 else 1 if x == 2 else 2 if x == 3 else 3, labels ) ) )
    
    #print( f"Buffer shape {buffer.shape} labels shape {labels.shape}" )
    buffer = np.transpose( buffer, (1, 0, 2) )
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
    
    for i,  modality  in enumerate(["EEG", "EOG", "EMG", "ECG"]):
        fp = np.memmap(
            f"{path}/{modality}_{subject_id}.dat", dtype="float32", mode="w+", shape= signal[i].shape
        )
        fp[:] = signal[i][:]
        fp.flush()
        del fp

    fp = np.memmap(
        f"{path}/y_{subject_id}.dat", dtype="int16", mode="w+", shape=labels.shape
    )
    
    fp[:] = labels[:]
    fp.flush()
    del fp
    return

# Specifica la directory in cui desideri scaricare i file
dl_dir =  get_data_folder()
dl_dir += "hpap/"
files = dl_dir + "physionet.org/files/ucddb/1.0.0/"

# check if the dataset exists

if not os.path.exists(files):
    logger.info("Fetching the dataset...")
    os.makedirs(dl_dir, exist_ok=True)
    # URL del dataset
    url = "https://anon.erda.au.dk/share_redirect/AtAGmjoATh"

    # Scarica il dataset
    subprocess.run(["wget", "-r", "-N", "-c", "-np", url, "-P", dl_dir])

    exit()
    
logger.info("Dataset is available at {}".format(files))

subject_details = pd.read_excel( files + "SubjectDetails.xls" )

table = pd.DataFrame()
table["subject_id"] = subject_details["S/No"]
table["file_id"] = subject_details["Study Number"]
table["gender"] = subject_details["Gender"]
table["age"] = subject_details["Age"]
#table["num_samples"] = subject_details["No of data blocks in EDF"]

logger.info("Saving table to csv...")

table.to_csv( dl_dir + "table.csv" )

num_samples = []

logger.info("Processing the data...")

os.makedirs( dl_dir + "raw", exist_ok = True )
os.makedirs( dl_dir + "xsleepnet", exist_ok = True )

raw_data, xsleepnet_data = [], []

# iter on the table rows using tqdm
for index, row in tqdm( table.iterrows(), total = table.shape[0] ):

    subject_id = row["subject_id"]
    filename = row["file_id"].lower()

    signal, labels = read_edf( files + filename )    
    num_samples.append( len(labels) )
    
    raw_data.extend( signal )
    
    xsleepnet = np.zeros( (signal.shape[0], 4, 29, 129) )
    for i in range(len(signal)):
        for m in range(4):    
            xsleepnet[i, m] = xsleepnet_preprocessing( signal[i, m] )

    xsleepnet_data.extend( xsleepnet )
    
    save_memmaps( dl_dir + "raw", signal, labels, subject_id )
    save_memmaps( dl_dir + "xsleepnet", xsleepnet, labels, subject_id )

table["num_samples"] = np.array(num_samples)
table.to_csv( dl_dir + "table.csv" )

logger.info("Data processing completed, computing standardization")

raw_data = np.array( raw_data )
xsleepnet_data = np.array( xsleepnet_data )

print(xsleepnet_data.shape, raw_data.shape)

raw_mean, raw_std = np.mean(raw_data, axis=0), np.std(raw_data, axis=0)
xsleepnet_mean, xsleepnet_std = np.mean(xsleepnet_data, axis=0), np.std(xsleepnet_data, axis=0)

logger.info( "Saving scaling parameters" )
# save the mean and std for each signal
np.savez(
    f"{dl_dir}/raw/scaling.npz",
    mean= raw_mean,
    std= raw_std,
)

np.savez(
    f"{dl_dir}/xsleepnet/scaling.npz",
    mean= xsleepnet_mean,
    std= xsleepnet_std,
)

print( raw_mean.shape, raw_std.shape, xsleepnet_mean.shape, xsleepnet_std.shape )

logger.info( "Saving splitting parameters" )
# computing the splitting subjects train valid test with ratio 0.7 0.15 0.15
# use a setted seed for reproducibility

np.random.seed(42)

train_subjects = np.random.choice( table["subject_id"], size = int( table.shape[0] * 0.7 ), replace = False )
valid_subjects = np.setdiff1d( table["subject_id"], train_subjects, assume_unique = True )
test_subjects = np.random.choice( valid_subjects, size = int( table.shape[0] * 0.15 ), replace = False )
valid_subjects = np.setdiff1d( valid_subjects, test_subjects, assume_unique = True )

print( train_subjects.shape, valid_subjects.shape, test_subjects.shape )

np.savez(
    f"{dl_dir}/splitting.npz",
    train = train_subjects,
    valid = valid_subjects,
    test = test_subjects
)