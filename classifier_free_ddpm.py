import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from Unets import ConditionalUnet
from utils import ddpm_schedules, load_MNIST

def train_diffusion(diffusion, device, n_epoch=50, sample_dir='log/classifier_free'):
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
            loss = diffusion(x, y)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        diffusion.eval()
        with torch.no_grad():
            y = (torch.arange(0, 40) % 10).to(device)
            xh = diffusion.sample(40, y, (1, 28, 28))
            grid = make_grid(xh, nrow=10)

            save_image(grid, f"{sample_dir}/ddpm_sample_{i}.png")

        # save diffusion model
        torch.save(diffusion.state_dict(), f"{sample_dir}/ddpm_mnist_{i}.pth")


class ClassifierFreeDDPM(nn.Module):
    def __init__( self, eps_model, betas, T, device, drop_prob=0.1):
        super(ClassifierFreeDDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.drop_prob = drop_prob

    def forward(self, x, y):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)  
        eps = torch.randn_like(x) 
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-self.alpha_bar[t, None, None, None]) * eps 

        context_mask = torch.bernoulli(torch.zeros_like(y)+self.drop_prob).to(self.device)

        return self.mse_loss(eps, self.eps_model(x_t, y, t / self.T, context_mask))

    def sample(self, n_sample, y, size, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_t = torch.randn(n_sample, *size).to(self.device) 

        # double the batch
        y_double = y.repeat(2)
        context_mask = torch.zeros_like(y_double).to(self.device)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_t_double = x_t.repeat(2,1,1,1)
            t_is_double = t_is.repeat(2,1,1,1)

            eps = self.eps_model(x_t_double, y_double, t_is_double, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2

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

    cfd = ClassifierFreeDDPM(eps_model=ConditionalUnet(in_channels=1), betas=(1e-4, 0.02), T=1000, device=device)
    train_diffusion(diffusion=cfd, device=device)
