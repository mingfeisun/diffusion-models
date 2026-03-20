import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, s, _ = x.shape
        x_reshaped = x.view(b, c, s * s).swapaxes(1, 2)
        x_ln = self.ln(x_reshaped)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x_reshaped = attn_out + x_reshaped
        x_reshaped = self.ff_self(x_reshaped) + x_reshaped
        return x_reshaped.swapaxes(2, 1).view(b, c, s, s)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )

    def forward(self, x):
        return self.conv(x)

class ConditionalUnet(nn.Module):
    def __init__(self, in_channels=3, n_classes=4, base_ch=128):
        super().__init__()
        # Encoder (downsampling)
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        
        # Bottleneck (8x8 size)
        self.bot1 = DoubleConv(base_ch * 4, base_ch * 4)
        self.attn = SelfAttention(base_ch * 4)
        self.bot2 = DoubleConv(base_ch * 4, base_ch * 4)

        # Decoder (upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = DoubleConv(base_ch * 6, base_ch * 2) 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = DoubleConv(base_ch * 3, base_ch)     
        self.outc = nn.Conv2d(base_ch, in_channels, kernel_size=1)

        # Conditional embedding
        self.label_emb = nn.Embedding(n_classes, base_ch * 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, base_ch * 4)
        )

    def forward(self, x, t, c):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.label_emb(c).unsqueeze(-1).unsqueeze(-1)
        emb = t_emb + c_emb

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x3 = self.bot1(x3 + emb)
        x3 = self.attn(x3)
        x3 = self.bot2(x3)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1) 
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) 
        x = self.up_conv2(x)
        return self.outc(x)

class IndoorClassifier(nn.Module):
    def __init__(self, in_ch=3, n_classes=4):
        super().__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        self.features = nn.Sequential(
            conv_block(in_ch, 64),    
            conv_block(64, 128),      
            conv_block(128, 256),     
            conv_block(256, 512),     
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x, t=None):
        feat = self.features(x).view(x.size(0), -1)
        return self.fc(feat)