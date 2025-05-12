import argparse
import pickle

import math
import os
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

from data.sentinel.dataset import create_dataloaders
from models.mnist_simple import MNISTDiffusion
from utils import ExponentialMovingAverage
from data.sentinel.make_synth_data import crop_and_resize
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
assert LambdaLR

def parse_args():
    parser = argparse.ArgumentParser(description="Training Sentinel`")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=64)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=196)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1_000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args
from datetime import datetime

def make_log_dir(base: str = "logs") -> str:
    """
    Create a sub-directory under `base` whose name is the current local
    timestamp YYYYMMDD-HHMMSS, e.g. logs/20250511-184237.

    Returns
    -------
    str
        The full path of the directory that was (or already) created.
    """
    # local time; use .astimezone(timezone.utc) if you prefer UTC
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base, ts)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def set_seed(seed):
    import os
    import random
    import numpy as np
    import torch

    # ------------ Python & NumPy ----------
    os.environ["PYTHONHASHSEED"] = str(seed)  # (helps with hash-based ops)
    random.seed(seed)
    np.random.seed(seed)

    # ------------ PyTorch -----------------
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPUs


def get_clean_grid(paths):
    images = []
    for path in paths:
        # crop 
        image = crop_and_resize(path, 112)
        image = image.resize((64, 64))
        images.append(np.array(image))
    return images
        

def main(args):
    set_seed(42)
    log_root = make_log_dir("/home/exx/Downloads/diffusion-cassini-logs/")
    
    print(f"Logging root is set to {log_root}")
    
    DATASET = "/home/exx/datasets/diffusion-cassini/sentinel/noised/v2"
    device="cpu" if args.cpu else "cuda"
    
    gt_lookup_path = os.path.join(DATASET, "gt.pkl")
    with open(gt_lookup_path, "rb") as f:
        gt_lookup = pickle.load(f)
        
    train_dataloader, test_dataloader=create_dataloaders(dataset_path=DATASET, clean_L=0, batch_size=args.batch_size, test_batch_size=args.n_samples, image_size=64, num_workers=1)
    model = MNISTDiffusion(timesteps=args.timesteps, image_size=64, in_channels=1, base_dim=args.model_base_dim, dim_mults=[1, 2, 4, 8]).to(
        device
    )
    
    eval_every = 25
    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)  # multiplier 1 → constant LR
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        
    test_images, test_Ls, test_paths = next(iter(test_dataloader))
    test_images = test_images.to(device)
    test_Ls = test_Ls.to(device)
    
    gt_images = get_clean_grid([gt_lookup[path] for path in test_paths])
    gt_images_noised = []
    for path in test_paths:
        image = Image.open(path)
        image = image.resize((64, 64))
        gt_images_noised.append(np.array(image))
    
    gt_images = [torchvision.transforms.ToTensor()(x) for x in gt_images]
    save_image(gt_images, f"{log_root}/gt.png", nrow=int(math.sqrt(args.n_samples)))

    gt_images_noised = [torchvision.transforms.ToTensor()(x) for x in gt_images_noised]
    save_image(gt_images_noised, f"{log_root}/gt_noised.png", nrow=int(math.sqrt(args.n_samples)))
    
    test_mask = test_Ls > 0
    test_noise_level_min = torch.zeros_like(test_Ls)
    test_noise_level_min[test_mask] = (model.timesteps / test_Ls[test_mask]).long()

    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,Ls, _) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            Ls = Ls.to(device)
            
            # higher L -> less noised. 
            # this says that we only sample more noise level than the image contains
            
            pred=model(image,noise, Ls=Ls)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        model_ema.eval()

        if (i+1) % eval_every == 0:
            samples = model_ema.module.sampling_Ls(
                test_images.clone(), 
                Ls=test_Ls,
                clipped_reverse_diffusion=not args.no_clip,
                device=device,
            )
            save_image(samples,log_root + "/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))
            save_image(test_images.clone(), log_root + "/steps_{:0>8}_gt.png".format(global_steps), nrow=int(math.sqrt(args.n_samples)))
    torch.save(ckpt, log_root + "/steps_{:0>8}.pt".format(global_steps))
                

if __name__=="__main__":
    args=parse_args()
    main(args)
