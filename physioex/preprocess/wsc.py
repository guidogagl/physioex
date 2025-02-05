import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
from loguru import logger
from scipy.signal import resample

from physioex.preprocess.preprocessor import Preprocessor
from physioex.preprocess.utils.signal import xsleepnet_preprocessing
from physioex.preprocess.utils.sleepdata import process_sleepdata_file


class WSCPreprocessor(Preprocessor):

    def __init__(
        self,
        visit: int = 1,
        preprocessors_name: List[str] = ["xsleepnet"],
        preprocessors=[xsleepnet_preprocessing],
        preprocessor_shape=[[3, 29, 129]],
        data_folder: str = None,
    ):

        assert visit in [1, 2, 3], "Visit must be 1, 2 or 3"

        self.visit = visit

        super().__init__(
            dataset_name=os.path.join("wsc", f"visit{visit}"),
            signal_shape=[3, 3000],
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def get_subjects_records(self) -> List[str]:

        records_dir = os.path.join(self.dataset_folder, "..", "raw_wsc", "polysomnography")

        records = os.listdir(records_dir)
        records = [
            os.path.join(records_dir, r)
            for r in records
            if (r.endswith(".edf") and f"visit{self.visit}" in r)
        ]

        # we need to add to each edf file the correspondig .stg.txt file
        records = [
            (r, r.replace(".edf", ".stg.txt"), r.replace(".edf", ".allscore.txt"))
            for r in records
        ]

        return records

    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:

        edf_path, stg_path, all_score_path = record

        # check if the allscore or stg file is present

        if os.path.exists(all_score_path):
            pass
            stages = read_allscore_file(all_score_path)
        elif os.path.exists(stg_path):
            pass
            stages = read_stg_file(stg_path)
        else:
            logger.error(f"Error: no stage file found for {record}")
            return None, None

        signal = read_edf_file(edf_path)

        if len(stages) - len(signal) == 1:
            stages = stages[:-1]

        if len(signal) != len(stages):

            logger.error(f"Error: signal and stages have different lengths")

            print(signal.shape, stages.shape)

            return None, None

        # check if the stages have unscored epochs

        unscored = np.where(stages == -1)[0]

        # remove unscored epochs from both signal and stages
        signal = np.delete(signal, unscored, axis=0)
        stages = np.delete(stages, unscored)

        if len(signal) == 0:
            logger.warning(f"Subject {record} has only unscored epochs")
            return None, None

        return signal, stages

    def customize_table(self, table) -> pd.DataFrame:

        return table


map_stages = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,
    "5": 4,
    "6": -1,
    "7": -1,
}


def read_stg_file(stg_file_path: str):

    with open(stg_file_path, "r") as f:
        lines = f.readlines()

    if lines[0] == "Epoch\tUser-Defined Stage\tCAST-Defined Stage\n":
        lines = lines[1:]

    # get the number only in each element of lines
    stages = []
    for line in lines:
        line = line.split("\t")

        stages.append([int(line[0]), int(line[1])])

    stages = np.array(stages)

    if stages[0, 0] > 0:
        stages[:, 0] -= stages[0, 0]

    # map the stages
    stages[:, 1] = np.array([map_stages[str(s)] for s in stages[:, 1]])

    return stages[:, 1]


import datetime


def read_allscore_file(all_score_path: str):

    with open(all_score_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    # get the START RECORDING LINE to get the start time of the recording

    start_time = None

    for line in lines:
        if "START RECORDING" in line:
            start_time = line.split("\t")[0]
            # convert the start time to a datetime object
            # start time example 23:33:23.00
            start_time = datetime.datetime.strptime(start_time, "%H:%M:%S.%f")
            break

    if start_time is None:
        logger.error("Error: start time not found in the file")
        return None

    stages = []
    current_time = start_time
    stage = 7  # unscored at the beginning
    for line in lines:
        if "STAGE - " in line:
            new_stage = line.split("STAGE - ")[-1].replace("\n", "")

            if new_stage == "W":
                new_stage = 0
            elif new_stage == "N1":
                new_stage = 1
            elif new_stage == "N2":
                new_stage = 2
            elif new_stage == "N3":
                new_stage = 3
            elif new_stage == "R":
                new_stage = 4
            elif new_stage == "NO STAGE" or new_stage == "MVT":
                new_stage = -1
            else:
                logger.error(f"Error: stage {new_stage} not recognized")
                return None

            stage_time = line.split("\t")[0]
            # format 23:33:23.00
            stage_time = datetime.datetime.strptime(stage_time, "%H:%M:%S.%f")
            # if the hours are 00 we need to add 1 day
            if stage_time.hour <= 12:
                stage_time += datetime.timedelta(days=1)

            # take the time passed from the last labelled epoch
            time_passed = stage_time - current_time

            # convert the time passed to seconds and add epochs to the stage list
            # approximating by excess
            passed_epochs = time_passed.total_seconds() / 30
            stages.extend([stage] * int(passed_epochs))

            stage = new_stage
            current_time = stage_time

    end_time = lines[-1].split("\t")[0]
    end_time = datetime.datetime.strptime(end_time, "%H:%M:%S.%f")
    end_time += datetime.timedelta(days=1)

    time_passed = end_time - current_time
    passed_epochs = time_passed.total_seconds() / 30
    stages.extend([stage] * int(passed_epochs))

    return np.array(stages)


import pyedflib

from scipy.signal import filtfilt, firwin, resample

fs = 100
Nfir = 100


def read_edf_file(edf_file_path: str):

    f = pyedflib.EdfReader(edf_file_path)

    # get all the channels available
    labels = f.getSignalLabels()

    eeg_label = ["C3_M2", "C3_M1", "C4_M1", "Fz_AVG", "C3_AVG"]
    try:
        eeg_index = [labels.index(l) for l in eeg_label if l in labels][0]
    except IndexError:
        logger.error(f"Error: no EEG channel found in {edf_file_path}")
        print(labels)
        exit()

    eeg = f.readSignal(eeg_index)

    old_fs = f.getSampleFrequency(eeg_index)

    # Creazione del filtro FIR bandpass
    b_band = firwin(Nfir + 1, [0.3, 40], pass_zero=False, fs=old_fs)
    # Applicazione del filtro al segnale EEG
    eeg = filtfilt(b_band, 1, eeg)

    if fs != old_fs:
        eeg = resample(eeg, int(len(eeg) * fs / old_fs))

    eog = f.readSignal(labels.index("E1")) - f.readSignal(labels.index("E2"))

    # filtering and resampling
    eog = filtfilt(b_band, 1, eog)

    if fs != old_fs:
        eog = resample(eog, int(len(eog) * fs / old_fs))

    chinlabel = ["chin", "cchin_l", "cchin_r", "rchin_l"]

    try:
        chinindex = [labels.index(l) for l in chinlabel if l in labels][0]
    except IndexError:
        logger.error(f"Error: no EMG channel found in {edf_file_path}")
        print(labels)
        exit()

    emg = f.readSignal(chinindex)

    b_band = firwin(Nfir + 1, 10, pass_zero=False, fs=old_fs)
    emg = filtfilt(b_band, 1, emg)

    if fs != old_fs:
        emg = resample(emg, int(len(emg) * fs / old_fs))

    # if the signal is not a multiple of  30 * fs, we need to add 0 padding at the beginning
    remainder = len(eeg) % (30 * fs)
    n_pads = (30 * fs) - remainder if remainder != 0 else 0

    if n_pads > 0:
        # add padding at the beginning of the sleep signal
        # logger.warning(f"Adding {n_pads} padding to the beginning of the signal")

        eeg = np.concatenate([np.zeros(n_pads), eeg])
        eog = np.concatenate([np.zeros(n_pads), eog])
        emg = np.concatenate([np.zeros(n_pads), emg])

    signal = np.array([eeg, eog, emg])
    signal = signal.reshape(3, len(eeg) // (30 * fs), 30 * fs)
    signal = signal.transpose(1, 0, 2)

    signal = signal[1:]

    return signal


if __name__ == "__main__":

    WSCPreprocessor(visit=1, data_folder="/mnt/vde/sleep-data/").run()
    WSCPreprocessor(visit=2, data_folder="/mnt/vde/sleep-data/").run()
    WSCPreprocessor(visit=3, data_folder="/mnt/vde/sleep-data/").run()
