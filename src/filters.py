import cv2

def apply_mean_filter(image, kernel_size=3):
    """Applies Mean Filter."""
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=3):
    """Applies Median Filter."""
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Applies Gaussian Filter."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_nlm_filter(image, h=10):
    """Applies Fast Non-Local Means Denoising."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
