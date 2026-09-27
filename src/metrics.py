import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def calculate_mse(imageA, imageB):
    """Calculates Mean Squared Error."""
    return mean_squared_error(imageA, imageB)

def calculate_psnr(imageA, imageB):
    """Calculates Peak Signal to Noise Ratio."""
    if np.array_equal(imageA, imageB):
        return float('inf')
    return peak_signal_noise_ratio(imageA, imageB, data_range=255)
