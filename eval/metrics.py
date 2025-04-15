import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """
    Compute the PSNR between two images.

    :param img1: First image in [H, W] or [H, W, C] format.
    :param img2: Second image in [H, W] or [H, W, C] format.
    :param max_pixel_value: The maximum pixel value of the images (e.g., 255 for 8-bit images).
    :return: PSNR value in decibels (dB).
    """
    # Ensure the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert to float for precision in MSE calculation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    # Mean Squared Error
    mse = np.mean((img1_float - img2_float) ** 2)
    if mse == 0:
        # Images are identical
        return float('inf')  # or a very large number, up to your preference.

    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr_value


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the SSIM between two images using scikit-image.

    :param img1: First image in [H, W] or [H, W, C] format.
    :param img2: Second image in [H, W] or [H, W, C] format.
    :return: SSIM value. 1 indicates perfect similarity.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # If images are color (e.g., shape [H, W, 3]), we can compute SSIM channel by channel
    # and average, or rely on scikit-image's built-in multichannel approach.
    if len(img1.shape) == 3 and img1.shape[2] in [3, 4]:
        # Use scikit-image's multichannel parameter
        ssim_value = ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())
    else:
        # Single-channel (grayscale) images
        ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())

    return ssim_value

def contrast_delta(image: np.ndarray, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> float:
    """
    Compute the contrast metric Δ as the difference between the
    upper_percentile and lower_percentile of pixel values.

    :param image: 2D numpy array (e.g., denoised log-magnitude image).
    :param lower_percentile: The 'low' percentile (default 2%).
    :param upper_percentile: The 'high' percentile (default 98%).
    :return: Contrast Δ = Q_{98%}(I) - Q_{2%}(I)
    """
    low_val = np.percentile(image, lower_percentile)
    high_val = np.percentile(image, upper_percentile)
    return float(high_val - low_val)

def edge_slope(denoised_image: np.ndarray, ground_truth_mask: np.ndarray, canny_thresh1: float = 0.0, canny_thresh2: float = 1.0) -> float:
    """
    Estimate edge preservation by measuring average pixel-to-pixel slopes
    around edges in the denoised image, using the ground_truth_mask to
    locate the object's edges.

    :param denoised_image: 2D numpy array of the denoised image (log scale).
    :param ground_truth_mask: 2D binary array (1=object, 0=background).
    :param canny_thresh1: Lower threshold for Canny edge detection on mask.
    :param canny_thresh2: Upper threshold for Canny edge detection on mask.
    :return: Average slope across all edge pixels.
    """
    # 1. Convert mask to uint8 so we can run Canny
    mask_uint8 = (ground_truth_mask > 0).astype(np.uint8) * 255

    # 2. Detect edges in the mask
    edges = cv2.Canny(mask_uint8, threshold1=canny_thresh1, threshold2=canny_thresh2)

    # 3. For each edge pixel, measure intensity difference to neighbors
    coords = np.argwhere(edges > 0)
    slopes = []
    for i, j in coords:
        val_center = denoised_image[i, j]
        # sample neighbor to the right
        if j + 1 < denoised_image.shape[1]:
            slopes.append(abs(denoised_image[i, j + 1] - val_center))
        # sample neighbor below
        if i + 1 < denoised_image.shape[0]:
            slopes.append(abs(denoised_image[i + 1, j] - val_center))

    if len(slopes) == 0:
        return 0.0

    return float(np.mean(slopes))
