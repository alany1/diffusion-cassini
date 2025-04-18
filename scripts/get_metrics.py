from importlib import import_module

import numpy as np
from PIL import Image
from params_proto import PrefixProto, Proto

from data.mnist.make_synth_data import create_ground_truth_digit
from eval.metrics import compute_psnr, compute_snr, compute_ssim, contrast_delta, edge_slope

from scipy.ndimage.measurements import variance

class EvalArgs(PrefixProto):
    # func = "models.lee_filter:lee_filter"
    func = "models.box_filter_geometric:box_filter_geometric"
    
    dataset_root = Proto(env="$DATASETS")
    dataset_prefix = "mnist/v2"
    
    seed = 42
    num_per_digit = 90
    font_size = 20
    image_size = (28, 28)
    font_path = "DejaVuSans-Bold.ttf"
    font_size = 20
    
def main(**deps):
    import random
    import os
    random.seed(EvalArgs.seed)
    
    EvalArgs._update(**deps)
    module_name, entrypoint = EvalArgs.func.split(":")
    module = import_module(module_name)
    model = getattr(module, entrypoint)
    
    digits = sorted(os.listdir(f"{EvalArgs.dataset_root}/{EvalArgs.dataset_prefix}/"))
    digits = [digit for digit in digits if digit != ".DS_Store"]

    all_metrics = []
    for digit in digits:
        digit_noised = [f for f in os.listdir(f"{EvalArgs.dataset_root}/{EvalArgs.dataset_prefix}/{digit}") if "None" not in f]

        
        # pick random images
        noisy_samples = random.sample(digit_noised, EvalArgs.num_per_digit)
        
        bg_colors = [float(x.split("bg_")[-1].split(".png")[0]) for x in noisy_samples]
        gt_digits = [create_ground_truth_digit(int(digit), image_size=EvalArgs.image_size, font_path=EvalArgs.font_path, font_size=EvalArgs.font_size, bg_color=bg_color) for bg_color in bg_colors]

        for i, (noisy_sample, gt_digit) in enumerate(zip(noisy_samples, gt_digits)):
            img = Image.open(f"{EvalArgs.dataset_root}/{EvalArgs.dataset_prefix}/{digit}/{noisy_sample}")
            img = np.array(img).astype(np.float32)
            
            # est = model(img, kernel_size = 3, sigma_noise=variance(img)) # compare Lee vs. ground truth
            est = model(img, kernel_size = 3) # compare box vs. ground truth
            # est = gt_digit # compare noisy vs. ground truth
            
            est = np.clip(est, 0, 255).astype(np.uint8)
            
            slope_mask = (est > bg_colors[i] * 255)
            
            all_metrics.append(dict(psnr=compute_psnr(est, gt_digit), snr=compute_snr(est, gt_digit), ssim=compute_ssim(est, gt_digit), contrast=contrast_delta(est), edge_slope=edge_slope(est, slope_mask)))
            
            
    # compute average across metrics
    avg_psnr = np.mean([x["psnr"] for x in all_metrics])
    avg_snr = np.mean([x["snr"] for x in all_metrics])
    avg_ssim = np.mean([x["ssim"] for x in all_metrics])
    avg_contrast = np.mean([x["contrast"] for x in all_metrics])
    avg_edge_slope = np.mean([x["edge_slope"] for x in all_metrics])
    
    print(f"Method {EvalArgs.func}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SNR: {avg_snr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Contrast: {avg_contrast:.4f}")
    print(f"Average Edge Slope: {avg_edge_slope:.4f}")
        
            
if __name__ == '__main__':
    main()
    
    

