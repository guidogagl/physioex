import math
import os
import re
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import filtfilt, firwin, resample

def read_edfrecords(filename, channel):
    # Apri il file EDF
    f = pyedflib.EdfReader(filename)

    channels = f.getSignalLabels()
    channel = channels.index(channel)

    # Ottieni la frequenza di campionamento
    fs = f.getSampleFrequency(channel)

    # Leggi il segnale dal canale specificato
    signal = f.readSignal(channel)

    # Chiudi il file EDF
    f.close()

    return signal, fs


def get_channels(filename):
    # Apri il file EDF
    f = pyedflib.EdfReader(filename)

    # Ottieni i nomi dei canali
    channels = f.getSignalLabels()

    # Chiudi il file EDF
    f.close()

    return channels


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def read_sleepdata_annotation(filename):
    
    def map_labels(row):
        # 0=Wake, 1=NREM, 2=REM, 3=Artifact

        if (row != 1) & (row != 2) & (row != 3):
            return 3 
        else:
            return int(row)-1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning) # ignore warning, it works properly
        df = pd.read_csv(filename, skiprows=9, engine='python', sep='\t', index_col=False)
    
    stages = df.iloc[:, 4] # get the stages column
    
    # code for filtering based on time. might be useful in the future.
    # zt_times = df.apply(lambda x: datetime.strptime(x['Time'][5:], '%H:%M:%S').time(), axis=1) # extract time from string
    # zt_times = zt_times.apply(lambda x: subtract_hours(x, 7)) # convert to ZT time
    # zt_times = zt_times.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second) # convert to seconds

    stages = stages.apply(map_labels)

    return stages

def process_sleepdata_file(edf_path, tsv_path):

    fs = 128
    epoch_second = 4

    # get the file name of the absolute path filename without the extension
    name = os.path.basename(edf_path)
    name = os.path.splitext(name)[0]

    available_channels = get_channels(edf_path)
    
    try:
        stages = read_sleepdata_annotation(tsv_path)
    except Exception as e:
        print(f"Error reading file: {tsv_path}")
        print(f"skipping subject")
        return None, None

    eeg1_channel = get_channel_from_available(available_channels, POSSIBLE_EEG1_CHANNELS)

    if eeg1_channel is None:
        print(f"Error: no EEG1 channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eeg1, old_fs = read_channel_signal(edf_path, eeg1_channel)

    # Parametri
    Nfir = 100

    # Creazione del filtro FIR bandpass
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

    # Applicazione del filtro al segnale EEG
    eeg1 = filtfilt(b_band, 1, eeg1)

    if fs != old_fs:
        eeg1 = resample(eeg1, int(len(eeg1) * fs / old_fs))

    eeg2_channel = get_channel_from_available(available_channels, POSSIBLE_EEG2_CHANNELS)
    if eeg2_channel is None:
        print(f"Error: no EEG2 channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eeg2, old_fs = read_channel_signal(edf_path, eeg2_channel)

    # filtering and resampling
    eeg2 = filtfilt(b_band, 1, eeg2)

    if fs != old_fs:
        eeg2 = resample(eeg2, int(len(eeg2) * fs / old_fs))

    emg_channel = get_channel_from_available(available_channels, POSSIBLE_EMG_CHANNELS)
    if emg_channel is None:
        print(f"Error: no EMG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        emg, old_fs = read_channel_signal(edf_path, emg_channel)

    # filtering and resampling
    b_band = firwin(Nfir + 1, 10, pass_zero=False, fs=old_fs)
    emg = filtfilt(b_band, 1, emg)

    if fs != old_fs:
        emg = resample(emg, int(len(emg) * fs / old_fs))

    expected_epochs = len(eeg1) // (epoch_second * fs)

    # checking coherence of the signals with stages
    if expected_epochs > len(stages):
        expected_epochs = len(stages)
    else:
        stages = stages[:expected_epochs]
    total_samples = expected_epochs * epoch_second * fs
    eeg1 = eeg1[:total_samples]
    eeg2 = eeg2[:total_samples]
    emg = emg[:total_samples]
    stages = np.array(stages)
    # print stages distribution
    # print(f'Stages distribution: {np.bincount(stages)}')

    # buffer the signals into epochs
    signal = np.array([eeg1, eeg2, emg])
    signal = np.transpose(signal).reshape(expected_epochs, epoch_second * fs, 3)

    # find the epochs associated with stages < 0 or > 2
    invalid_epochs = np.where(np.logical_or(stages < 0, stages > 2))[0]

    # remove the invalid epochs
    stages = np.delete(stages, invalid_epochs)
    signal = np.delete(signal, invalid_epochs, axis=0)
    
    signal = np.transpose(signal, (0, 2, 1))

    return signal.astype(np.float32), stages.astype(int)

def read_channel_signal(filename, channel):

    signal, old_fs = read_edfrecords(filename, channel)

    return signal, old_fs


def get_channel_from_available(available_channels, possible_channels):
    # return the first element of possible_channels which is into available_channels
    # if possible channels [i] is a tuple, then all the elements of the tuple must be in available_channels

    # the selection must be case insensitive and return the original channel name
    av_channels = [ch.upper() for ch in available_channels]

    for channel in possible_channels:
        if isinstance(channel, tuple):
            if all([ch in av_channels for ch in channel]):
                return (
                    available_channels[av_channels.index(channel[0])],
                    available_channels[av_channels.index(channel[1])],
                )
        else:
            if channel in av_channels:
                return available_channels[av_channels.index(channel)]


POSSIBLE_EEG1_CHANNELS = [
    "EEG",
    "EEG1",
    "EEG EEG1A-B",
]

POSSIBLE_EEG2_CHANNELS = [
    "EEG2",
    "EEG EEG2A-B",
]

POSSIBLE_EMG_CHANNELS = [
    "EMG",
    "EMG EMG",
]
