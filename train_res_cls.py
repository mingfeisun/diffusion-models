import torch, os, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from res_models import ResNetIndoorClassifier

class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, root="/autodl-fs/data/diffusion-models-main/data/cleaned_dataset/"):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.classes = ["livingroom", "bedroom", "corridor", "kitchen"]
        self.imgs, self.labels = [], []
        for i, c in enumerate(self.classes):
            p = os.path.join(root, c)
            for f in os.listdir(p):
                if not f.startswith('.'):
                    self.imgs.append(os.path.join(p, f))
                    self.labels.append(i)

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.transform(Image.open(self.imgs[i]).convert("RGB")), self.labels[i]

def train_classifier():
    device = torch.device("cuda")
    loader = DataLoader(ConditionalDataset(), batch_size=64, shuffle=True, num_workers=4)
    model = ResNetIndoorClassifier().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n🚀 启动 ResNet 导航员抗噪训练...")
    
    for epoch in range(80):
        model.train()
        current_max_t = min(0.1 + (epoch / 50) * 0.7, 0.8)
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Noise:{current_max_t:.2f}]")
        
        correct, total = 0, 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            # 模拟随机加噪
            t = torch.rand(x.size(0), 1, 1, 1).to(device) * current_max_t
            x_noisy = torch.sqrt(1-t) * x + torch.sqrt(t) * torch.randn_like(x)
            
            logits = model(x_noisy)
            loss = criterion(logits, y)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            acc = (logits.argmax(1) == y).float().mean()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
        
        if (correct/total) > 0.94 and epoch > 30: 
            print("✅ 表现优异，提前收工！")
            break

    torch.save(model.state_dict(), "resnet_classifier_best.pth")
    print("⭐ 导航员已就位：resnet_classifier_best.pth")

if __name__ == "__main__":
    train_classifier()