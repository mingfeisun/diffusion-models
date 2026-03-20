import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os
import numpy as np

from guideUnets import IndoorClassifier

class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, root="/autodl-fs/data/diffusion-models-main/data/cleaned_dataset/"):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.classes = ["livingroom", "bedroom", "corridor", "kitchen"]
        self.imgs, self.labels = [], []
        
        for i, c in enumerate(self.classes):
            p = os.path.join(root, c)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if not f.startswith('.'):
                        self.imgs.append(os.path.join(p, f))
                        self.labels.append(i)
        print(f"✅ Loaded {len(self.imgs)} images, classifier ready!")

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): 
        return self.transform(Image.open(self.imgs[i]).convert("RGB")), self.labels[i]

def train_classifier():
    device = torch.device("cuda")
    ds = ConditionalDataset()
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    
    model = IndoorClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n🚀 Start enhanced classifier training (Acc target: 85%+)")
    best_acc = 0.0

    for epoch in range(100):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        current_max_t = min(0.1 + (epoch / 50) * 0.6, 0.7)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/100 [MaxNoise: {current_max_t:.2f}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            t = torch.rand(x.size(0), 1).to(device) * current_max_t
            noise = torch.randn_like(x)
            alpha_bar = (1.0 - t).view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
            
            logits = model(x_noisy, t)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            total_loss += loss.item()
            
            acc = 100 * correct / total
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
        
        scheduler.step()
        epoch_acc = 100 * correct / total
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "indoor_classifier_best.pth")
            print(f"⭐ New best accuracy: {best_acc:.2f}%")

        if epoch_acc >= 85.0 and epoch > 30:
            print(f"✅ Accuracy target reached, early stopping!")
            break

    print(f"🏁 Training finished, best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_classifier()