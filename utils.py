import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta        = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha       = 1 - beta
    alpha_bar   = torch.cumsum(torch.log(alpha), dim=0).exp()
    sigma       = torch.sqrt(beta)

    return {
        "alpha": alpha,  # \alpha_t
        "alpha_bar": alpha_bar,  # \bar{\alpha_t}
        "beta": beta, # \beta (will be used as \sigma_t^2)
        "sigma": sigma, # \sigma
    }

def load_MNIST():
    tf = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)

    return dataloader