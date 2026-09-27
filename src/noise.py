from skimage.util import random_noise
import numpy as np

def add_gaussian_noise(image, var=0.01):
    """Adds Gaussian noise to an image."""
    noisy = random_noise(image, mode='gaussian', var=var)
    return (255 * noisy).astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.05):
    """Adds Salt and Pepper noise to an image."""
    noisy = random_noise(image, mode='s&p', amount=amount)
    return (255 * noisy).astype(np.uint8)
