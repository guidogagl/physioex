import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
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
    # print(f'Reading file: {filename}')
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except Exception as e:
        print(f"Error reading file: {filename}")
        raise e

    stages = []
    starts = []
    durations = []

    for event in root.findall(".//ScoredEvent"):
        event_type = event.find("EventType").text
        if event_type == "Stages|Stages":
            event_concept = event.find("EventConcept").text
            start = float(event.find("Start").text)
            duration = float(event.find("Duration").text)
            stage = int(event_concept.split("|")[-1])

            assert duration % 30 == 0
            num_epoch = int(duration / 30)
            stages.extend([stage] * num_epoch)
            starts.append(start)
            durations.append(duration)

    # Verifica che non ci siano errori nel numero di epoche
    assert (starts[-1] + durations[-1]) / 30 == len(
        stages
    ), "Error: mismatch in number of epochs in annotations file"

    # print(f'Number of stages: {len(stages)}')

    return stages


def process_sleepdata_file(edf_path, xml_path):

    ret = 1

    fs = 100
    epoch_second = 30
    win_size = 2
    overlap = 1
    nfft = next_power_of_2(win_size * fs)

    # print(nfft)

    # get the file name of the absolute path filename without the extension
    name = os.path.basename(edf_path)
    name = os.path.splitext(name)[0]

    # stages = read_sleepdata_annotation( os.path.join( xml_path, f'{name}-nsrr.xml'))

    try:
        stages = read_sleepdata_annotation(xml_path)
    except Exception as e:
        print(f"Error reading file: {xml_path}")
        print(f"skipping subject")
        return None, None
    available_channels = get_channels(edf_path)

    eeg_channel = get_channel_from_available(available_channels, POSSIBLE_EEG_CHANNELS)

    if eeg_channel is None:
        print(f"Error: no EEG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eeg, old_fs = read_channel_signal(edf_path, eeg_channel)

    # Parametri
    Nfir = 100

    # Creazione del filtro FIR bandpass
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)

    # Applicazione del filtro al segnale EEG
    eeg = filtfilt(b_band, 1, eeg)

    if fs != old_fs:
        eeg = resample(eeg, int(len(eeg) * fs / old_fs))

    eog_channel = get_channel_from_available(available_channels, POSSIBLE_EOG_CHANNELS)
    if eog_channel is None:
        print(f"Error: no EOG channel found in {edf_path}")
        print(f"Available channels: {available_channels}")
        return None, None
    else:
        eog, old_fs = read_channel_signal(edf_path, eog_channel)

    # filtering and resampling
    eog = filtfilt(b_band, 1, eog)

    if fs != old_fs:
        eog = resample(eog, int(len(eog) * fs / old_fs))

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

    expected_epochs = len(eeg) // (epoch_second * fs)

    # checking coherence of the signals with stages
    if expected_epochs > len(stages):
        expected_epochs = len(stages)
    else:
        stages = stages[:expected_epochs]
    total_samples = expected_epochs * epoch_second * fs
    eeg = eeg[:total_samples]
    eog = eog[:total_samples]
    emg = emg[:total_samples]
    stages = np.array(stages)
    # print stages distribution
    # print(f'Stages distribution: {np.bincount(stages)}')

    # buffer the signals into epochs
    signal = np.array([eeg, eog, emg])
    signal = np.transpose(signal).reshape(expected_epochs, epoch_second * fs, 3)

    # find the epochs associated with stages < 0 or > 5
    invalid_epochs = np.where(np.logical_or(stages < 0, stages > 5))[0]

    # remove the invalid epochs
    stages = np.delete(stages, invalid_epochs)
    signal = np.delete(signal, invalid_epochs, axis=0)

    # print(f'Invalid epochs: {len(invalid_epochs)}')
    # print(f'Valid epochs: {len(stages)}')

    # remove Wake epochs if Wake is the biggest class:
    count_stage = np.bincount(stages)
    if count_stage[0] > max(count_stage[1:]):  # if too much W
        # print('Wake is the biggest class. Trimming it..')
        second_largest = max(count_stage[1:])

        W_ind = stages == 0  # W indices
        last_evening_W_index = np.where(np.diff(W_ind) != 0)[0][0] + 1
        if stages[0] == 0:  # only true if the first epoch is W
            num_evening_W = last_evening_W_index
        else:
            num_evening_W = 0

        first_morning_W_index = np.where(np.diff(W_ind) != 0)[0][-1] + 1
        num_morning_W = len(stages) - first_morning_W_index + 1

        nb_pre_post_sleep_wake_eps = num_evening_W + num_morning_W
        if nb_pre_post_sleep_wake_eps > second_largest:
            total_W_to_remove = nb_pre_post_sleep_wake_eps - second_largest
            if num_evening_W > total_W_to_remove:
                stages = stages[total_W_to_remove:]
                signal = signal[total_W_to_remove:]
            else:
                evening_W_to_remove = num_evening_W
                morning_W_to_remove = total_W_to_remove - evening_W_to_remove
                stages = stages[evening_W_to_remove : len(stages) - morning_W_to_remove]
                signal = signal[evening_W_to_remove : len(signal) - morning_W_to_remove]

        # print(f'New stages distribution: {np.bincount(stages)}')
    else:
        # print('Wake is not the biggest class, nothing to remove.')
        pass
    stages_from = [0, 1, 2, 3, 4, 5]
    stages_to = [0, 1, 2, 3, 3, 4]

    # map stages to the new stages
    stages = np.array([stages_to[s] for s in stages])
    signal = np.transpose(signal, (0, 2, 1))

    # print(signal.shape, stages.shape)

    return signal.astype(np.float32), stages.astype(int)


def read_channel_signal(filename, channel):

    if isinstance(channel, tuple):
        c4, old_fs = read_edfrecords(filename, channel[0])
        m1, old_fs = read_edfrecords(filename, channel[1])
        eeg = c4 - m1
    else:
        eeg, old_fs = read_edfrecords(filename, channel)

    return eeg, old_fs


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


POSSIBLE_EEG_CHANNELS = [
    "EEG(sec)",
    "EEG",
    "EEG1",
    ("C4", "M1"),
    ("C4", "A1"),
    "C4-M1",
    "C4-A1",
    "C4M1",
    "C4A1",
]

POSSIBLE_EOG_CHANNELS = [
    ("EOG(L)", "EOG(R)"),
    ("EOGL", "EOGR"),
    ("EOG-L", "EOG-R"),
    ("E1", "E2"),
    ("E1-M2", "E2-M1"),
    ("LOC", "ROC"),
    ("E-1", "E-2"),
    ("L-EOG", "R-EOG"),
    "E1-E2",
    ("E2-M1", "E2-M1"),
]

POSSIBLE_EMG_CHANNELS = [
    "EMG",
    ("LCHIN", "CCHIN"),
    ("LCHIN", "RCHIN"),
    ("L CHIN", "R CHIN"),
    ("L CHIN", "C CHIN"),
    ("CHIN1", "CHIN2"),
    "CHIN",
    "CHIN EMG",
    "EMG CHIN",
    "CHIN1-CHIN2",
    "LCHIN-CCHIN",
    ("EMG3", "EMG1"),
    ("EMG3", "EMG2"),
    "EMG3-EMG1",
    "EMG3-EMG2",
    "EMG1-EMG3",
    "EMG",
    ("L-LEGS", "R-LEGS"),
]
