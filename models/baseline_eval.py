import torch
import numpy as np
from torchvision import transforms

from data.mnist.dataset import create_eval_dataloader
from models.mnist_simple import MNISTDiffusion
from utils import ExponentialMovingAverage
from PIL import Image

class args:
    timesteps = 1_000
    model_base_dim = 128
    model_ema_steps = 10
    epochs = 100
    device = 'cuda'
    batch_size = 128
    model_ema_decay = 0.995
    ckpt = "/home/exx/mit/diffusion-cassini/results_baseline/steps_00008514.pt"
    test_dataset = "/home/exx/datasets/diffusion-cassini/mnist/brian_v2"
    
    preprocess = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  # scales image to [0,1]
                transforms.Lambda(lambda x: torch.log1p(x)),  # apply log(1+x) transform
                transforms.Normalize([0.3863], [0.1982]),  # suggested normalization after log1p
            ]
        )
    
    model_ema = None
    
def setup():
    model = MNISTDiffusion(timesteps=args.timesteps, image_size=28, in_channels=1, base_dim=args.model_base_dim, dim_mults=[2, 4]).to(
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
    test_dataloader = create_eval_dataloader(dataset_path="/home/exx/datasets/diffusion-cassini/mnist/v2", clean_L=0,  test_batch_size=1, image_size=28, num_workers=1)
    model = setup()
    for i, (test_images, test_Ls) in enumerate(test_dataloader):
        run_step(model, test_images, test_Ls)
