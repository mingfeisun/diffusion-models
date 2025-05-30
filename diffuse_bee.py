import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import matplotlib.image as mpimg

from tqdm import tqdm

from Unets import Unet
from utils import ddpm_schedules, load_UoM

class DDPM(nn.Module):
    def __init__( self, eps_model, betas, T, device):
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v.to(device))

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)  
        eps = torch.randn_like(x) 
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-self.alpha_bar[t, None, None, None]) * eps 

        return self.mse_loss(eps, self.eps_model(x_t, t / self.T)) # normalize t

    def forward_diffusion(self, x):
        all_x = []
        for t in range(self.T):
            eps = torch.randn_like(x) 
            x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-self.alpha_bar[t, None, None, None]) * eps 
            all_x.append(x_t)

        return all_x

    def sample(self, n_sample, size):
        x_t = torch.randn(n_sample, *size).to(self.device) 

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample,1,1,1)
            eps = self.eps_model(x_t, t_is)
            x_t = 1/torch.sqrt(self.alpha[t]) * (x_t - eps * (1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t]) ) + self.sigma[t] * z

        return x_t

if __name__ == "__main__":
    if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
        device_type = "cuda"
    elif torch.backends.mps.is_available(): # i.e. for Apple Silicon GPUs
        device_type = "mps" 
    else:
        device_type = "cpu"
    
    device = torch.device(device_type) # Select best available device

    ddpm = DDPM(eps_model=Unet(in_channels=3), betas=(1e-4, 0.02), T=1000, device=device)

    with torch.no_grad():
        tf = transforms.Compose( [transforms.ToTensor(), 
                                transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0)), 
                                ])
        image = mpimg.imread('bee.jpg')[:, :, :3]
        image = tf(image).to(device)

        diffused_samples = ddpm.forward_diffusion(image[None])
        selected_samples = []
        for i, sample in enumerate(diffused_samples[::80]):
            selected_samples.append(sample[0] + 0.5)  # unnormalize
        grid = make_grid(selected_samples[:8], nrow=len(selected_samples))
        save_image(grid, f"diffused_bee.png")