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


def xsleepnet_preprocessing(signals, preprocessor_shape, fs=100, window="hamming", nperseg=200, noverlap=100, nfft=256):

    num_windows, n_channels, n_timestamps = signals.shape

    # transform each signal into its spectrogram ( fast )
    # nfft 256, noverlap 1, win 2, fs 100, hamming window

    S = np.zeros((num_windows, n_channels, preprocessor_shape[1], preprocessor_shape[2])).astype(np.float32)

    for i in range(num_windows):
        for j in range(n_channels):

            _, _, Sxx = spectrogram(
                signals[i, j].astype(np.double).reshape(-1),
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
            )

            # log_10 scale the spectrogram safely (using epsilon)
            Sxx = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

            Sxx = np.transpose(Sxx, (1, 0))

            S[i, j] = Sxx.astype(np.float32)

    return S.astype(np.float32)


def xsleepnet_preprocessing_mouse(signals, preprocessor_shape):
    
    return xsleepnet_preprocessing(signals,
                                   preprocessor_shape,
                                   fs=128,
                                   window="hamming",
                                   nperseg=256,
                                   noverlap=240,
                                   nfft=256)
    

class OnlineVariance:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.shape = shape

    def add(self, data):
        # data.shape = batch_size, data_shape
        for x in data:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (x - self.mean)

    def compute(self):
        if self.n < 2:
            return np.zeros(self.shape).astype(np.float32), np.zeros(self.shape).astype(
                np.float32
            )

        variance = self.M2 / (self.n - 1)
        variance = np.reshape(variance, self.shape)
        mean = np.reshape(self.mean, self.shape)

        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
