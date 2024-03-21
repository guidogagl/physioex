import numpy as np
import torch
from scipy import signal as signal

from torchaudio.transforms import Spectrogram


######### time frequency image transform from X SleepNet Article ##########
x_fs = 100
x_win_size = 2
x_overlap = 1
x_nfft = 256
x_spectogram = Spectrogram(
    n_fft=x_nfft,
    win_length=x_win_size * x_fs,
    hop_length=x_overlap * x_fs,
    window_fn= torch.hamming_window,
)

def xsleepnet_transform(x, x_spectogram = x_spectogram):
    x = x_spectogram(x)
    
    x = 20 * torch.log10(x + torch.finfo(torch.float32).eps)
    
    #x = x.permute(0, 1, 3, 2)
    return x
 
 ###########################################################################