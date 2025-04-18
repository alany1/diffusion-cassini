import argparse
import math
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

from data.mnist.dataset import create_dataloaders
from models.mnist_simple import MNISTDiffusion
from utils import ExponentialMovingAverage


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=128)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1_000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    # device="cpu"
    train_dataloader,test_dataloader=create_dataloaders(dataset_path="/home/exx/datasets/diffusion-cassini/mnist/v2", clean_L=0, batch_size=args.batch_size, test_batch_size=args.n_samples, image_size=28, num_workers=1)
    model = MNISTDiffusion(timesteps=args.timesteps, image_size=28, in_channels=1, base_dim=args.model_base_dim, dim_mults=[2, 4]).to(
        device
    )
    
    L_max = 64
    eval_every = 1
    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        
    test_images, test_Ls = next(iter(test_dataloader))
    test_images = test_images.to(device)
    test_Ls = test_Ls.to(device)

    test_mask = test_Ls > 0
    test_noise_level_min = torch.zeros_like(test_Ls)
    test_noise_level_min[test_mask] = (model.timesteps / test_Ls[test_mask]).long()

    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,Ls) in enumerate(train_dataloader):
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

        os.makedirs("results",exist_ok=True)
        # torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        # image, Ls = next(iter(test_dataloader))
        # image=image.to(device)
        # Ls = Ls.to(device)
        # # higher L -> less noised. 
        # # this says that we only sample more noise level than the image contains
        # mask = Ls > 0
        # noise_level_min = torch.zeros_like(Ls)
        # noise_level_min[mask] = (model.timesteps / Ls[mask]).long()
        # samples = model_ema.module.sampling_starting_from_noise_level(test_images, noise_level_min=test_noise_level_min, clipped_reverse_diffusion=not args.no_clip, device=device)
        
        if (i+1) % eval_every == 0:
            samples = model_ema.module.sampling_Ls(
                test_images.clone(), 
                Ls=test_Ls,
                clipped_reverse_diffusion=not args.no_clip,
                device=device,
            )
            save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))
                

if __name__=="__main__":
    args=parse_args()
    main(args)
