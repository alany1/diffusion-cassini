import glob
from importlib import import_module
from tqdm import tqdm
import numpy as np
from PIL import Image
from params_proto import PrefixProto, Proto
from tqdm import tqdm
from data.mnist.make_synth_data import create_ground_truth_digit
from data.sentinel.make_synth_data import crop_and_resize
from eval.metrics import compute_psnr, compute_snr, compute_ssim, contrast_delta, edge_slope

from scipy.ndimage.measurements import variance

class EvalArgs(PrefixProto):
    # func = "models.lee_filter:lee_filter"
    # func = "models.box_filter_geometric:box_filter_geometric"
    func = "models.sentinel_eval:eval_func"
    
    dataset_root = Proto(env="$DATASETS")
    dataset_prefix = "mnist/test/v0"
    
    seed = 42
    num_per_class = 10
    image_size = (64, 64)
    
    viz = False
    
def main(**deps):
    import random
    import os
    import pickle
    EvalArgs._update(**deps)
    
    random.seed(EvalArgs.seed)
    
    module_name, entrypoint = EvalArgs.func.split(":")
    module = import_module(module_name)
    model = getattr(module, entrypoint)
    
    all_metrics = []
    
    with open(f"{EvalArgs.dataset_root}/{EvalArgs.dataset_prefix}/gt.pkl", "rb") as f:
        gt = pickle.load(f)

    all_images = glob.glob(os.path.join(EvalArgs.dataset_root, EvalArgs.dataset_prefix, "**/*.png"), recursive=True)
    
    test_images = []
    gt_images = []
    for cls in ["agri", "barrenland",  "grassland", "urban"]:
        cls_images = [f for f in all_images if cls in f]
        t = random.sample(cls_images, EvalArgs.num_per_class)
        g = [gt[f] for f in cls_images if f in gt]
        
        test_images.extend(t)
        gt_images.extend(g)
        
    for path in tqdm(test_images, desc="evaluating"):
        noisy_sample = Image.open(path)
        noisy_sample = np.array(noisy_sample).astype(np.float32)
        L_value = int(path.split("L_")[-1].split("_")[0])
        
        est = model(noisy_sample, L=L_value, kernel_size = 3, sigma_noise=variance(noisy_sample))  # compare Lee vs. ground truth
        est = np.clip(est, 0, 255).astype(np.uint8)
        est_norm = est / 255.0
        # slope_mask = est > bg_colors[i] * 255
        
        gt_path = gt[path]
        gt_image = crop_and_resize(gt_path, 112)
        gt_image = gt_image.resize((64, 64))
        gt_image = np.array(gt_image)
        
        # side by side of gt and est
        if EvalArgs.viz:
            from matplotlib import pyplot as plt
            plt.subplot(1, 3, 1)
            plt.imshow(gt_image, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(noisy_sample, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(est, cmap='gray')
            plt.show()
        
        all_metrics.append(
            dict(
                psnr=compute_psnr(est, gt_image),
                snr=compute_snr(est, gt_image),
                ssim=compute_ssim(est, gt_image),
                contrast=contrast_delta(gt_image),
                # edge_slope=edge_slope(est_norm, slope_mask),
            )
        )

    avg_psnr = np.mean([x["psnr"] for x in all_metrics])
    avg_snr = np.mean([x["snr"] for x in all_metrics])
    avg_ssim = np.mean([x["ssim"] for x in all_metrics])
    avg_contrast = np.mean([x["contrast"] for x in all_metrics])
    
    print(f"Method {EvalArgs.func}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SNR: {avg_snr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Contrast: {avg_contrast:.4f}")
        
            
if __name__ == '__main__':
    main(dataset_prefix="sentinel/test/v2",
         # func="models.lee_filter:lee_filter",)
        # func = "models.box_filter_geometric:box_filter_geometric",)
         # func="models.baseline_eval:eval_func",)
        func="models.sentinel_eval:eval_func",)
    
    

