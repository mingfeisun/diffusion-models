import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

from tqdm import tqdm

from Unets import ConditionalUnet
from utils import ddpm_schedules, load_MNIST

from Unets import Unet
from ddpm import DDPM

def classifier_grad_fn(x, classifier, y, _scale=1):
    """
    return the graident of the classifier outputing y wrt x.
    formally expressed as d_log(classifier(x, t)) / dx
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        grad = torch.autograd.grad(selected.sum(), x_in)[0] * _scale
        return grad

class GuidedDDPM(nn.Module):
    def __init__( self, eps_model, betas, T, device, classifier, _grad_fn, _scale=1.0):
        super(GuidedDDPM, self).__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device

        # guided sampling
        self.classifier = classifier
        self._grad_fn = _grad_fn
        self._scale = _scale

    def forward(self, x):
        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)  
        eps = torch.randn_like(x) 
        x_t = torch.sqrt(self.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-self.alpha_bar[t, None, None, None]) * eps 

        return self.mse_loss(eps, self.eps_model(x_t, t / self.T))

    def sample(self, n_sample, y, size):
        x_t = torch.randn(n_sample, *size).to(self.device) 

        for t in reversed(range(self.T)):
            z = torch.randn(n_sample, *size).to(self.device) if t > 1 else 0
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample,1,1,1)
            eps = self.eps_model(x_t, t_is)

            # guidance
            x_t_mean = 1/torch.sqrt(self.alpha[t]) * (x_t - eps * (1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t]) )
            x_t_variance = self.sigma[t]
            gradient = self._grad_fn(x_t_mean, self.classifier, y, self._scale)
            x_t_mean = x_t_mean + x_t_variance * gradient

            x_t = x_t_mean + self.sigma[t] * z

        return x_t

def train_classifier(classifier, diffusion, device, n_epoch=10):
    print('########## Train classifier ##########')
    classifier.to(device)

    dataloader = load_MNIST()
    optim = torch.optim.Adam(classifier.parameters(), lr=2e-4)
    ce_loss = nn.CrossEntropyLoss()

    def pred_accuracy(preds, labels):
        y = torch.max(preds, 1)[1]
        return torch.mean(y.eq(labels).float())

    save_interval = 5

    classifier_dir = 'log/classifier'
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    for i in range(n_epoch):
        classifier.train()

        pbar = tqdm(dataloader)
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            t = torch.randint(1, diffusion.T, (x.shape[0],)).to(device)  
            eps = torch.randn_like(x) 
            x_t = torch.sqrt(diffusion.alpha_bar[t, None, None, None]) * x + torch.sqrt(1-diffusion.alpha_bar[t, None, None, None]) * eps 

            preds = classifier(x_t)

            loss = ce_loss(preds, y) 
            loss.backward()
            acc = pred_accuracy(preds, y)

            pbar.set_description(f"loss: {loss:.4f}, acc: {acc:.3f}")
            optim.step()

        # save diffusion model
        if (i+1) % save_interval == 0:
            torch.save(classifier.state_dict(), f"{classifier_dir}/classifier_mnist_{i}.pth")

def guided_sampling(device):
    ddpm = DDPM(eps_model=Unet(in_channels=1), betas=(1e-4, 0.02), T=1000, device=device)
    ddpm_ckpt_path = 'log/samples/ddpm_mnist_45.pth'
    ddpm.load_state_dict(torch.load(ddpm_ckpt_path))
    eps_model = ddpm.eps_model
    eps_model.to(device)

    ## build classifier
    from Unets import EncoderUnet
    classifier = EncoderUnet(in_channels=1)
    classifier.to(device)

    ## guided ddpm
    g_ddpm = GuidedDDPM(eps_model=eps_model, betas=(1e-4, 0.02), T=1000, device=device, 
                        classifier=classifier, _grad_fn=classifier_grad_fn, _scale=1.0)
    g_ddpm.to(device)

    train_classifier(classifier, g_ddpm, device)
    ## load a check point
    # classifier_ckpt_path = 'log/classifier/classifier_mnist_9.pth'
    # classifier.load_state_dict(torch.load(classifier_ckpt_path))

    sample_dir = 'log/samples_guided'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    g_ddpm.eval()
    with torch.no_grad():
        y = (torch.arange(0, 40) % 10).to(device)
        xh = g_ddpm.sample(40, y, (1, 28, 28))
        grid = make_grid(xh, nrow=10)

        save_image(grid, f"{sample_dir}/guided_sample.png")


if __name__ == "__main__":
    if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
        device_type = "cuda"
    elif torch.backends.mps.is_available(): # i.e. for Apple Silicon GPUs
        device_type = "mps" 
    else:
        device_type = "cpu"
    
    device = torch.device(device_type) # Select best available device
    guided_sampling(device)
