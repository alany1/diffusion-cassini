import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.measurements import variance
from PIL import Image
import os

import matplotlib.pyplot as plt

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

def box_filter(img, kernel_size=7):
    return uniform_filter(img, size=kernel_size)

def box_filter_geometric(img, kernel_size=7):
    log_img = np.log(img + 1e-12)
    log_mean = uniform_filter(log_img, size=kernel_size)
    return np.exp(log_mean)

def box_filter_geometric_padded(img, kernel_size=7, pad_mode='reflect'):
    pad = kernel_size // 2
    pad_width = ((pad, pad), (pad, pad))
    img_padded = np.pad(img, pad_width, mode=pad_mode)

    log_img = np.log(img_padded + 1e-12)
    log_mean = uniform_filter(log_img, size=kernel_size, mode='constant')
    log_mean = log_mean[pad:-pad, pad:-pad]
    return np.exp(log_mean)

if __name__=="__main__":
    # -------------
    # MNIST Testing
    # -------------
    digit = 8
    folder_path = f"/Users/brianli/Desktop/diffusion_models/datasets/mnist/brian/v0/{digit}/"

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        img_path = folder_path + file
        parsed = file.split("_")

        print("parsed", parsed)
        if parsed[3] == 'None':
            continue
        print("L:", int(parsed[3]))
        print("variance (parsed):", 1/int(parsed[3]))

        img = Image.open(img_path)
        img_np = np.array(img).astype(np.float32)

        sigma_noise = variance(img_np)
        print("variance (calc):", sigma_noise)

        # filtered_img = lee_filter(img_np, kernel_size=3, sigma_noise=sigma_noise)
        # filtered_img = box_filter(img_np, kernel_size=3)
        filtered_img = box_filter_geometric_padded(img_np, kernel_size=3)

        diff_img = img_np - filtered_img


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Filtered Image (Lee Filter)")
        plt.imshow(filtered_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Diff")
        plt.imshow(diff_img, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # --------------------
    # Single Image Testing
    # --------------------
        
    # img_path = "/Users/brianli/Desktop/diffusion_models/datasets/lagoon.jpg"
    # img = Image.open(img_path)
    # img_np = np.array(img).astype(np.float32) / 255.0 

    # sigma_noise = variance(img_np)
    # print("variance:", sigma_noise)
    # filtered_img = lee_filter(img_np, kernel_size=15, sigma_noise=sigma_noise)

    # diff_img = img_np - filtered_img


    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(img_np, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Filtered Image (Lee Filter)")
    # plt.imshow(filtered_img, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Diff")
    # plt.imshow(diff_img, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()



        
    

    # img_path = "test_image.jpg"  # Replace with your image path
    # img = Image.open(img_path).convert("L")  # Convert to grayscale

    # # === Step 2: Convert to NumPy array and normalize ===
    # img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # # === Step 3: Apply Lee filter ===
    # filtered = lee_filter(img_np, kernel_size=7, sigma_noise=0.01)

    # # === Step 4: Plot the original and filtered image ===
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(img_np, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.title("Filtered Image (Lee Filter)")
    # plt.imshow(filtered, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()