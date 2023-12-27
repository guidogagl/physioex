from scipy import signal as signal
import numpy as np
import torch

def xsleepnet_transform(x, fs = 100, win_size = 2, overlap = 1, nfft = 256):
    x_transf = torch.zeros( x.shape[0], x.shape[1], 29, 129).float()
    
    # loop into the channels
    for i in range(x.shape[1]):
        # loop into the samples
        for j in range(x.shape[0]):
            f, t, Sxx = signal.spectrogram(x[j,i].numpy(), fs = fs, window="hamming", nperseg = win_size * fs, noverlap = overlap * fs, nfft = nfft)
            Sxx = 20*np.log10(Sxx + np.finfo(float).eps)

            x_transf[j, i] = torch.tensor(np.transpose(Sxx)).float()
            
    return  x_transf