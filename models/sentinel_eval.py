import torch
import numpy as np
from torchvision import transforms

from models.mnist_simple import MNISTDiffusion
from utils import ExponentialMovingAverage, crop_2x2_grid, crop_4x4_grid, stitch_4x4_grid, stitch_2x2_grid
from PIL import Image

class args:
    timesteps = 1_000
    model_base_dim = 196
    model_ema_steps = 10
    epochs = 100
    device = 'cuda'
    batch_size = 64
    model_ema_decay = 0.995
    # new
    ckpt = "/home/exx/Downloads/diffusion-cassini-logs/20250511-202703/steps_00012500.pt"
    adjust_gamma = True
    # old method, but scaled up
    # ckpt = "/home/exx/Downloads/diffusion-cassini-logs/20250511-214322/steps_00012500.pt"
    # adjust_gamma = False
    test_dataset = "/home/exx/datasets/diffusion-cassini/sentinel/test/v0"
    
    preprocess = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),  # scales image to [0,1]
                transforms.Lambda(lambda x: torch.log1p(x)),  # apply log(1+x) transform
                transforms.Normalize([0.3863], [0.1982]),  # suggested normalization after log1p
            ]
        )
    
    model_ema = None
    
def setup():
    model = MNISTDiffusion(timesteps=args.timesteps, image_size=64, in_channels=1, base_dim=args.model_base_dim, dim_mults=[1, 2, 4, 8], adjust_gamma=args.adjust_gamma).to(
        args.device
    )
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=args.device, decay=1.0 - alpha)

    ckpt = torch.load(args.ckpt)
    model_ema.load_state_dict(ckpt["model_ema"])
    model.load_state_dict(ckpt["model"])

    model_ema.eval()
    return model_ema

def run_step(model_ema, test_images, test_Ls):
    test_images = test_images.to(args.device)
    test_Ls = test_Ls.to(args.device)

    samples = model_ema.module.sampling_Ls(
        test_images.clone(),
        Ls=test_Ls,
        device=args.device,
    )
    
    img = samples[0][0].cpu()
    img = (img.clip(0, 1)* 255).numpy().astype(np.uint8)
    
    return img

def eval_func(img_og, L, **kwargs):
    """
    Warning, image is coming as float32 0-255, not good. fix ASAP. 
    """
    if args.model_ema is None:
        args.model_ema = setup()
        
    img_og = Image.fromarray(img_og.astype(np.uint8))
    img = img_og.convert("L")
    img = args.preprocess(img)
    
    out = args.model_ema.module.sampling_Ls(
        img[None, ...].clone(),
        Ls=torch.tensor([L]).long(),
        device=args.device,
    )
    result = out[0][0].cpu()
    result = (result.clip(0, 1) * 255).numpy().astype(np.uint8)
    
    return result
    

if __name__ == '__main__':
    from utils import crop_8x8_grid, crop_4x4_grid, stitch_8x8_grid, stitch_4x4_grid
        
    from matplotlib import pyplot as plt
    import os
    model = setup()
    img1 = Image.open("/home/exx/datasets/diffusion-cassini/Cassini_T3-Large/IMG0195_L42.png")
    img2 = Image.open("/home/exx/datasets/diffusion-cassini/Cassini_T3-Large/IMG0118_L17.png")
    img3 = Image.open("/home/exx/datasets/diffusion-cassini/Cassini_T3-Large/IMG0202_L41.png")
    
    imgs = [img1, img2, img3] 
    Ls = [42, 17, 41]
    keys = ["IMG0195_L42", "IMG0118_L17", "IMG0202_L41"]
    
    save_root = "/home/exx/Downloads/cassini_results_new/"
    os.makedirs(save_root, exist_ok=True)
      
    def run_patch_experiment(img_og, L, key):
        save_dir = os.path.join(save_root, key)
        os.makedirs(save_dir, exist_ok=True)
        
        img_og = np.array(img_og)[..., :3]
        
        
        
        # out_patches = []
        # patches = crop_8x8_grid(img_og)
        # for patch in patches:
        #     out = eval_func(patch, L)
        #     out_patches.append(out)
        #     
        # out = stitch_8x8_grid(out_patches)
        # out = Image.fromarray(out)
        # plt.imshow(out, cmap="gray"); plt.show()
        # out.save(os.path.join(save_dir, f"8x8_{L}.png"))
        # 
        # 
        out_patches = []
        patches = crop_4x4_grid(img_og)
        for patch in patches:
            out = eval_func(patch, L)
            out_patches.append(out)

        out = stitch_4x4_grid(out_patches)
        plt.imshow(out, cmap="gray")
        plt.show()
        out = Image.fromarray(out)
        out.save(os.path.join(save_dir, f"4x4_{L}.png"))
    
        out_patches = []
        patches = crop_2x2_grid(img_og)
        for patch in patches:
            out = eval_func(patch, L)
            out_patches.append(out)

        out = stitch_2x2_grid(out_patches)
        out = Image.fromarray(out)
        plt.imshow(out, cmap="gray")
        plt.show()
        out.save(os.path.join(save_dir, f"2x2_{L}.png"))
            
        out_patches = []
        out = eval_func(img_og, L)
        out = Image.fromarray(out)
        plt.imshow(out, cmap="gray")
        plt.show()
        out.save(os.path.join(save_dir, f"full_{L}.png"))
        
        print('done')
        
    for img, L, key in zip(imgs, Ls, keys):
        run_patch_experiment(img, L, key)
        
        
        
