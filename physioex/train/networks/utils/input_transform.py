import numpy as np
import torch

######### time frequency image transform from X SleepNet Article ##########

def xsleepnet_transform(x):
    fs = 100
    win_size = 2
    overlapping = 1
    nfft = 256

    x_shape = x.size()
    x = x.view(-1, x_shape[-1])

    hop_length = (x_shape[-1] - int(fs * win_size)) // 29

    x = torch.stft(
        x,
        window=torch.hamming_window(window_length=int(fs * win_size)),
        n_fft=nfft,
        hop_length=hop_length,
        win_length=int(fs * win_size),
        return_complex=False,
    )[..., 0] # only the real part
    x = x[:, :129, :29]
    x = x.permute(0, 2, 1)
    x = x.reshape(*x_shape[:-1], 29, 129)

    return x


###########################################################################
