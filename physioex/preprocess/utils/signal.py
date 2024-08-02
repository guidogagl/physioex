import os

import numpy as np
from scipy.signal import butter, filtfilt, resample, spectrogram


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def xsleepnet_preprocessing(signals):

    num_windows, n_channels, n_timestamps = signals.shape

    # transform each signal into its spectrogram ( fast )
    # nfft 256, noverlap 1, win 2, fs 100, hamming window

    S = np.zeros((num_windows, n_channels, 29, 129)).astype(np.float32)

    for i in range(num_windows):
        for j in range(n_channels):

            _, _, Sxx = spectrogram(
                signals[i, j].astype(np.double).reshape(-1),
                fs=100,
                window="hamming",
                nperseg=200,
                noverlap=100,
                nfft=256,
            )

            # log_10 scale the spectrogram safely (using epsilon)
            Sxx = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

            Sxx = np.transpose(Sxx, (1, 0))

            S[i, j] = Sxx.astype(np.float32)

    return S.astype(np.float32)


class online_variance:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.shape = -1

    def set_shape(self, shape):
        self.shape = shape
        return self

    def add(self, data):

        for x in data:
            x = np.reshape(x, -1).astype(np.double)

            self.n += 1
            delta = x - self.mean
            self.mean = self.mean + delta / self.n
            self.M2 = self.M2 + delta * (x - self.mean)

    def compute(self):
        variance = self.M2 / (self.n - 1)

        variance = np.reshape(variance, self.shape)
        mean = np.reshape(self.mean, self.shape)

        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
