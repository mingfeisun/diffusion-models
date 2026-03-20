import torch

def ddpm_schedules(beta1, beta2, T, device="cuda"):
    beta = torch.linspace(beta1, beta2, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return {
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "one_over_sqrt_alpha": 1.0 / torch.sqrt(alpha),
        "sqrt_beta": torch.sqrt(beta),
    }