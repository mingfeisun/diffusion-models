import os, torch, time, copy, csv, shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import numpy as np
from torch_fidelity import calculate_metrics
from res_models import ConditionalUnet, ResNetIndoorClassifier
from res_utils import ddpm_schedules
from train_res_cls import ConditionalDataset 

class EMA:
    def __init__(self, beta=0.999): self.beta = beta
    def update(self, ma_model, current_model):
        for p, ma_p in zip(current_model.parameters(), ma_model.parameters()):
            ma_p.data = ma_p.data * self.beta + p.data * (1 - self.beta)

def classifier_grad_fn(x, classifier, y, scale):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        grad = torch.autograd.grad(selected.sum(), x_in)[0]
        return grad * scale

def evaluate_fid(model, classifier, device, epoch, real_path, sched):
    print(f"\n🔔 [Epoch {epoch}] 启动 Guided FID 采样 (64张)...")
    temp_dir = f"./temp_fid_{epoch}/"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    model.eval(); classifier.eval()
    with torch.no_grad():
        for b in range(2):
            y = torch.arange(4).repeat_interleave(8).to(device) 
            x_gen = torch.randn(32, 3, 64, 64).to(device)
            for i in range(1000, 0, -1):
                t_in = (torch.full((32,), i, device=device).float()/1000).view(-1,1)
                
                grad = classifier_grad_fn(x_gen, classifier, y, scale=2.5)
                eps = model(x_gen, t_in, y) - torch.sqrt(1-sched["alpha_bar"][i-1]) * grad
                
                x_gen = sched["one_over_sqrt_alpha"][i-1] * (x_gen - (1-sched["alpha"][i-1])/sched["sqrt_one_minus_alpha_bar"][i-1] * eps)
                if i > 1: x_gen += sched["sqrt_beta"][i-1] * torch.randn_like(x_gen)
            
            for j in range(32):
                img = (x_gen[j] * 0.5 + 0.5).clamp(0, 1)
                utils.save_image(img, os.path.join(temp_dir, f"fid_{b}_{j}.png"))
    
    try:
        metrics = calculate_metrics(input1=temp_dir, input2=real_path, cuda=True, samples_find_deep=True, fid=True, verbose=False)
        return metrics['frechet_inception_distance']
    except: return 999.0
    finally: shutil.rmtree(temp_dir)

def train():
    device = torch.device("cuda")
    run_name = f"ResNet_Guided_Final_{time.strftime('%m%d_%H%M')}"
    save_dir = f"./log/{run_name}/"
    os.makedirs(save_dir, exist_ok=True)

    unet = ConditionalUnet(base_ch=128).to(device)
    ema_model = copy.deepcopy(unet).eval()
    ema = EMA(beta=0.995) 

    # 加载特训好的ResNet
    classifier = ResNetIndoorClassifier().to(device)
    ckpt = "resnet_classifier_best.pth"
    if os.path.exists(ckpt):
        classifier.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"✅ 导航员已就位: {ckpt}")
    classifier.eval()

    ds = ConditionalDataset()
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1.2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2600, eta_min=1e-5)
    sched = ddpm_schedules(1e-4, 0.02, 1000, device)
    
    real_path = "/autodl-fs/data/diffusion-models-main/data/cleaned_dataset/" 

    log_file = os.path.join(save_dir, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "fid"])

    for epoch in range(1, 2601):
        unet.train()
        loss_epoch = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, c in pbar:
            x, c = x.to(device), c.to(device)
            t = torch.randint(1, 1001, (x.shape[0],), device=device)
            eps = torch.randn_like(x)
            alpha_bar = sched["alpha_bar"][t-1].view(-1,1,1,1)
            x_t = torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*eps
            
            eps_theta = unet(x_t, (t.float()/1000).view(-1,1), c)
            loss = F.mse_loss(eps, eps_theta)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ema.update(ema_model, unet)
            loss_epoch.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(loss_epoch):.4f}")
        
        scheduler.step()

        if epoch % 100 == 0:
            ema_model.eval()
            with torch.no_grad():
                y = torch.arange(4).repeat_interleave(4).to(device)
                x_gen = torch.randn(16, 3, 64, 64).to(device)
                for i in range(1000, 0, -1):
                    t_in = (torch.full((16,), i, device=device).float()/1000).view(-1,1)
                    cur_scale = 1.5 + 4.5 * (i / 1000)
                    grad = classifier_grad_fn(x_gen, classifier, y, cur_scale)
                    eps_guided = ema_model(x_gen, t_in, y) - torch.sqrt(1-sched["alpha_bar"][i-1]) * grad
                    x_gen = sched["one_over_sqrt_alpha"][i-1] * (x_gen - (1-sched["alpha"][i-1])/sched["sqrt_one_minus_alpha_bar"][i-1] * eps_guided)
                    if i > 1: x_gen += sched["sqrt_beta"][i-1] * torch.randn_like(x_gen)
                
                x_vis = x_gen.clone()
                for b in range(16):
                    x_vis[b] = (x_vis[b] - x_vis[b].min()) / (x_vis[b].max() - x_vis[b].min() + 1e-5)
                utils.save_image(x_vis, f"{save_dir}/epoch_{epoch}.png", nrow=4)
                torch.save(ema_model.state_dict(), f"{save_dir}/unet_ema_{epoch}.pth")

        if epoch % 200 == 0:
            fid = evaluate_fid(ema_model, classifier, device, epoch, real_path, sched)
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch, np.mean(loss_epoch), fid])

if __name__ == "__main__":
    train()