import torch
import torch.nn as nn
import torchvision.models as models

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=512, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        emb_out = self.emb_layer(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out
        h = self.conv2(h)
        return h + self.shortcut(x)

class ConditionalUnet(nn.Module):
    def __init__(self, in_channels=3, n_classes=4, base_ch=128):
        super().__init__()
        emb_dim = base_ch * 4
        
        self.label_emb = nn.Embedding(n_classes, emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.inc = ResBlock(in_channels, base_ch, emb_dim)
        self.down1 = nn.MaxPool2d(2)
        self.res_down1 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.down2 = nn.MaxPool2d(2)
        self.res_down2 = ResBlock(base_ch * 2, base_ch * 4, emb_dim)
        
        self.bot1 = ResBlock(base_ch * 4, base_ch * 4, emb_dim)
        self.bot2 = ResBlock(base_ch * 4, base_ch * 4, emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_up1 = ResBlock(base_ch * 6, base_ch * 2, emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_up2 = ResBlock(base_ch * 3, base_ch, emb_dim)
        
        self.outc = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x, t, c):
        emb = self.time_mlp(t) + self.label_emb(c)

        x1 = self.inc(x, emb)
        x2 = self.res_down1(self.down1(x1), emb)
        x3 = self.res_down2(self.down2(x2), emb)
        
        x3 = self.bot1(x3, emb)
        x3 = self.bot2(x3, emb)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1) 
        x = self.res_up1(x, emb)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) 
        x = self.res_up2(x, emb)
        return self.outc(x)

class ResNetIndoorClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity() 
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        return self.model(x)