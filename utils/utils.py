import torch
import numpy as np

#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def crop_8x8_grid(img_512):
    """Return list of 64 numpy arrays shaped (64, 64, C)."""
    patches = []
    for y in range(0, 512, 64):  # 0, 64, … 448
        for x in range(0, 512, 64):
            patches.append(img_512[y : y + 64, x : x + 64])
    return patches  # len == 64

def crop_4x4_grid(img_512):
    """Return list of 64 numpy arrays shaped (64, 64, C)."""
    patches = []
    for y in range(0, 512, 128):  # 0, 64, … 448
        for x in range(0, 512, 128):
            patches.append(img_512[y : y + 128, x : x + 128])
    return patches  # len == 64

def crop_2x2_grid(img_512):
    """Return list of 64 numpy arrays shaped (64, 64, C)."""
    patches = []
    for y in range(0, 512, 256):  # 0, 64, … 448
        for x in range(0, 512, 256):
            patches.append(img_512[y : y + 256, x : x + 256])
    return patches  # len == 64

def stitch_8x8_grid(patches):
    """
    patches – list/array length 64, each (64,64[,C])
              assumed order: (row0-col0, row0-col1, … row7-col7)
    returns  – single 512×512(×C) image
    """
    if isinstance(patches, np.ndarray) and patches.ndim == 5:
        # optional: convert a 5-D tensor (8,8,64,64,C) back — rare
        patches = patches.reshape(-1, *patches.shape[-3:])

    # build each row by concatenating 8 patches side-by-side
    rows = [
        np.concatenate(patches[i * 8 : (i + 1) * 8], axis=1)  # axis 1 ≡ x-direction
        for i in range(8)
    ]
    whole = np.concatenate(rows, axis=0)  # axis 0 ≡ y-direction
    return whole

def stitch_4x4_grid(patches):
    """
    patches – list/array length 64, each (64,64[,C])
              assumed order: (row0-col0, row0-col1, … row7-col7)
    returns  – single 512×512(×C) image
    """
    if isinstance(patches, np.ndarray) and patches.ndim == 5:
        # optional: convert a 5-D tensor (8,8,64,64,C) back — rare
        patches = patches.reshape(-1, *patches.shape[-3:])

    # build each row by concatenating 8 patches side-by-side
    rows = [
        np.concatenate(patches[i * 4 : (i + 1) * 4], axis=1)  # axis 1 ≡ x-direction
        for i in range(4)
    ]
    whole = np.concatenate(rows, axis=0)
    return whole

def stitch_2x2_grid(patches):
    """
    patches – list/array length 64, each (64,64[,C])
              assumed order: (row0-col0, row0-col1, … row7-col7)
    returns  – single 512×512(×C) image
    """
    if isinstance(patches, np.ndarray) and patches.ndim == 5:
        # optional: convert a 5-D tensor (8,8,64,64,C) back — rare
        patches = patches.reshape(-1, *patches.shape[-3:])

    # build each row by concatenating 8 patches side-by-side
    rows = [
        np.concatenate(patches[i * 2 : (i + 1) * 2], axis=1)  # axis 1 ≡ x-direction
        for i in range(2)
    ]
    whole = np.concatenate(rows, axis=0)
    return whole
