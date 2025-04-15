import numpy as np
from scipy.ndimage import uniform_filter


def lee_filter(img, kernel_size=7, sigma_noise=0.1):
    """
    Works by computing local statistics (mean and variance) and determines how much of original image to keep
    vs. an averaged version

    params:
        kernel_size (int): 
        sigma_noise (float): 

    """
    local_mean = uniform_filter(img, size=kernel_size)
    local_mean_sq = uniform_filter(img**2, size=kernel_size)
    local_variance = local_mean_sq - local_mean**2

    weights = (local_variance - sigma_noise) / (local_variance + 1e-8)
    weights = np.clip(weights, 0, 1)

    filtered_img = local_mean + weights * (img - local_mean)

    return filtered_img
