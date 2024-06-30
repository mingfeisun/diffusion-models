import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from Unets import Unet
from utils import ddpm_schedules, load_MNIST

class DDPM(nn.Module):
    def __init__( self, eps_model, betas, T, device):
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)  
        eps = torch.randn_like(x) 
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-self.alpha_bar[t, None, None, None]) * eps 

        return self.mse_loss(eps, self.eps_model(x_t, t / self.T)) # normalize t

    def sample(self, n_sample, size):
        x_t = torch.randn(n_sample, *size).to(self.device) 

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample,1,1,1)
            eps = self.eps_model(x_t, t_is)
            x_t = 1/torch.sqrt(self.alpha[t]) * (x_t - eps * (1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t]) ) + self.sigma[t] * z

        return x_t

def train_diffusion(diffusion, device, n_epoch=50, sample_dir='log/samples'):
    diffusion.to(device)

    dataloader = load_MNIST()
    optim = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir) 

    for i in range(n_epoch):
        diffusion.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = diffusion(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        diffusion.eval()
        with torch.no_grad():
            xh = diffusion.sample(16, (1, 28, 28))
            grid = make_grid(xh, nrow=4)

            save_image(grid, f"{sample_dir}/ddpm_sample_{i}.png")

        # save diffusion model
        torch.save(diffusion.state_dict(), f"{sample_dir}/ddpm_mnist_{i}.pth")

if __name__ == "__main__":
    if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
        device_type = "cuda"
    elif torch.backends.mps.is_available(): # i.e. for Apple Silicon GPUs
        device_type = "mps" 
    else:
        device_type = "cpu"
    
    device = torch.device(device_type) # Select best available device

    ddpm = DDPM(eps_model=Unet(in_channels=1), betas=(1e-4, 0.02), T=1000, device=device)
    train_diffusion(diffusion=ddpm, device=device)