import math
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Unet
from utils.bucket_schedule import make_betas


class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8], adjust_gamma=False):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size
        if adjust_gamma:
            betas = make_betas()
        else:
            betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self, x, noise, Ls=None):
        # x: NCHW
        if Ls is None:
            # Sample uniformly from [0, self.timesteps)
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        else:
            # noise_level_min is assumed to be a tensor of shape (B, 1).
            # For each sample, we want to sample an integer in
            # [noise_level_min[i], self.timesteps). We can do this by:
            # 1. Generate a uniform random number u in [0,1) for each sample.
            # 2. Multiply by the range length: (self.timesteps - noise_level_min[i])
            # 3. Floor the result to get an integer offset, then add noise_level_min.
            # noise_level_min=Ls
            # u = torch.rand(x.shape[0], device=x.device)
            # range_length = (self.timesteps - noise_level_min).to(x.device)  # tensor of shape (B,)
            # t = (noise_level_min + torch.floor(u * range_length)).long()
            # t = t.clamp(max=self.timesteps - 1)
            t_per_L = self.timesteps / 65
            _L = torch.arange(1, 65, device=x.device).long()
            L_to_noise_level = self.timesteps - _L*t_per_L
            # make 1‑element, 1‑D tensors on the same device & dtype
            pad_lo = torch.tensor([0], device=x.device, dtype=torch.long)
            pad_hi = torch.tensor([1000], device=x.device, dtype=torch.long)

            # concatenate along the single dimension
            L_to_noise_level = torch.cat([pad_lo, L_to_noise_level, pad_hi], dim=0).long()

            noise_level_min = L_to_noise_level[Ls]
            
            next_Ls = Ls - 1
            next_Ls[next_Ls < 0] = 64
            next_Ls[next_Ls == 0] = 65 
            
            noise_level_max = L_to_noise_level[next_Ls]
            u = torch.rand(x.shape[0], device=x.device)

            range_length = (noise_level_max - noise_level_min).to(x.device)  # tensor of shape (B,)
            t = (noise_level_min + torch.floor(u * range_length)).long()
            # t -= 1
            t = t.clamp(max=self.timesteps - 1)

        x_t = self._forward_diffusion(x, t, noise, noise_level_min=noise_level_min)
        pred_noise = self.model(x_t, t)

        return pred_noise
    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"): 
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t

    @torch.no_grad()
    def sampling_starting_from_noise_level(self, start_samples, noise_level_min, clipped_reverse_diffusion=True, device="cuda"):
        """
        start_samples: tensor of shape (B, C, H, W) assumed to be at timestep noise_level_min.
        noise_level_min: either a single integer (or 0-dim tensor) or a tensor of shape (B, 1)

        Returns the denoised images after reversing the diffusion process from noise_level_min down to 0.
        """
        x = start_samples.to(device)
        B = x.shape[0]

        # Case 1: noise_level_min is a single integer (or scalar tensor)
        if isinstance(noise_level_min, int) or (torch.is_tensor(noise_level_min) and noise_level_min.dim() == 0):
            t_start = int(noise_level_min)
            # Reverse diffusion from t_start-1 down to 0 (for all samples simultaneously)
            for i in tqdm(range(t_start - 1, -1, -1), desc="Sampling (uniform noise_level_min)"):
                t = torch.full((B,), i, device=device)
                noise = torch.randn_like(x)
                if clipped_reverse_diffusion:
                    x = self._reverse_diffusion_with_clip(x, t, noise)
                else:
                    x = self._reverse_diffusion(x, t, noise)
        else:
            # Case 2: noise_level_min is a tensor of shape (B,1)
            # For each sample, run its reverse process individually.
            for b in tqdm(range(B), desc="Sampling (per-sample noise_level_min)"):
                t_start = int(noise_level_min[b].item())
                for i in range(t_start - 1, -1, -1):
                    t_val = torch.tensor([i], device=device)
                    noise = torch.randn_like(x[b : b + 1])
                    if clipped_reverse_diffusion:
                        x[b : b + 1] = self._reverse_diffusion_with_clip(x[b : b + 1], t_val, noise)
                    else:
                        x[b : b + 1] = self._reverse_diffusion(x[b : b + 1], t_val, noise)

        # Map from [-1,1] to [0,1]
        x = self.invert_preprocess(x)
        return x
    

    @torch.no_grad()
    def sampling_Ls(self, start_samples, Ls, clipped_reverse_diffusion=True, device="cuda", silent=True):
        """
        start_samples: tensor of shape (B, C, H, W) assumed to be at timestep noise_level_min.
        noise_level_min: either a single integer (or 0-dim tensor) or a tensor of shape (B, 1)

        Returns the denoised images after reversing the diffusion process from noise_level_min down to 0.
        """

        t_per_L = self.timesteps / 65
        _L = torch.arange(1, 65, device=device).long()
        L_to_noise_level = self.timesteps - _L * t_per_L
        # make 1‑element, 1‑D tensors on the same device & dtype
        pad_lo = torch.tensor([0], device=device, dtype=torch.long)
        pad_hi = torch.tensor([1000], device=device, dtype=torch.long)

        # concatenate along the single dimension
        L_to_noise_level = torch.cat([pad_lo, L_to_noise_level, pad_hi], dim=0).long()
        
        x = start_samples.to(device)
        B = x.shape[0]
        
        noise_levels = L_to_noise_level[Ls]
        
        it = tqdm(range(B), desc="Sampling (per-sample noise_level_min)", disable=silent)
        for b in it:
            t_start = int(noise_levels[b].item())
            path = range(t_start - 1, -1, -1)
            for i in path:
                t_val = torch.tensor([i], device=device)
                noise = torch.randn_like(x[b : b + 1])
                x[b : b + 1] = self._reverse_step_local_eps(x[b : b + 1], t_val, noise)
                # if clipped_reverse_diffusion:
                #     x[b : b + 1] = self._reverse_diffusion_with_clip(x[b : b + 1], t_val, noise)
                # else:
                #     x[b : b + 1] = self._reverse_diffusion(x[b : b + 1], t_val, noise)

        # Map from [-1,1] to [0,1]
        x = self.invert_preprocess(x)
        return x

    def invert_preprocess(self, x, mean=0.3863, std=0.1982):
        # Denormalize
        x = x * std + mean
        # Invert log1p: log1p(x) = log(1+x)  so  x = expm1(log1p(x))
        x = torch.expm1(x)
        # Ensure values are in [0,1]
        x = torch.clamp(x, 0, 1)
        return x
        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise, noise_level_min=None):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        if noise_level_min is None:
            return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                    self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
        else:
            fact = self.alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) / self.alphas_cumprod.gather(-1, noise_level_min).reshape(x_0.shape[0], 1, 1, 1)
            return torch.sqrt(fact) * x_0 + torch.sqrt(1-fact)*noise  

    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise, clip_bounds=(-1.95, 1.55)): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(*clip_bounds)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise

    @torch.no_grad()
    def _reverse_step_local_eps(self, x_t, t, noise):
        """
        One DDPM reverse step when the network predicts *local* ε (t‑1→t).

        Args:
            x_t   : current latent (B,C,H,W)
            t     : timestep tensor (B,)  int64
            noise : N(0,1)  same shape as x_t
        Returns:
            x_{t-1} sample
        """
        assert t.shape == torch.Size([1])
        
        eps_hat = self.model(x_t, t)                                # ε̂_local

        beta_t  = self.betas.gather(0, t).view(-1,1,1,1)
        alpha_t = self.alphas.gather(0, t).view(-1,1,1,1)
        alpha_bar_t   = self.alphas_cumprod.gather(0, t).view(-1,1,1,1)

        mu = (x_t - torch.sqrt(beta_t) * eps_hat) / torch.sqrt(alpha_t)
        
        if t.item() == 0:
            return mu

        alpha_bar_tm1 = self.alphas_cumprod.gather(0, t-1).clamp(min=1e-20).view(-1,1,1,1)
        # analytic posterior variance
        var = beta_t * (1. - alpha_bar_tm1) / (1. - alpha_bar_t)
        std = torch.sqrt(var)
        # if any t == 0, just return the mean (clean sample)
        std = torch.where(t.view(-1,1,1,1) == 0, torch.zeros_like(std), std)


        return mu + std * noise
