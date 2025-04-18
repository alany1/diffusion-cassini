import numpy as np
from scipy.ndimage import uniform_filter

def box_filter_geometric(img, kernel_size=7, pad_mode='reflect', **_):
    pad = kernel_size // 2
    pad_width = ((pad, pad), (pad, pad))
    img_padded = np.pad(img, pad_width, mode=pad_mode)

    log_img = np.log(img_padded + 1e-12)
    log_mean = uniform_filter(log_img, size=kernel_size, mode='constant')
    log_mean = log_mean[pad:-pad, pad:-pad]
    return np.exp(log_mean)
