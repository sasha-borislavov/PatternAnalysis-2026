"""Models for ADNI AD vs NC classification (PyTorch only, no NumPy).

BaselineCNN: small 4-block convolutional network used as the comparison
baseline for the confidence/calibration investigation. GFNet is added later.
"""
import torch.nn as nn


class BaselineCNN(nn.Module):
    """4 x (conv3x3 -> BatchNorm -> ReLU -> maxpool) -> global avg pool -> linear."""

    def __init__(self, num_classes=2, in_ch=1, widths=(16, 32, 64, 128)):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class GFNetBlock(nn.Module):
    """norm -> frequency filter -> norm -> MLP, with one residual connection."""

    def __init__(self, dim, h, w, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        return x + self.mlp(self.norm2(self.filter(self.norm1(x))))


class GFNet(nn.Module):
    """Small Global Filter Network: patch embedding, filter blocks, pooled linear head."""

    def __init__(self, num_classes=2, in_ch=1, img_size=224, patch=16,
                 dim=128, depth=4, mlp_ratio=4, drop=0.1):
        super().__init__()
        g = img_size // patch  # token grid is g x g (14 x 14 for 224 / 16)
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.Sequential(*[GFNetBlock(dim, g, g, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (batch, tokens, dim)
        x = self.drop(x + self.pos)
        x = self.norm(self.blocks(x)).mean(dim=1)           # average over tokens
        return self.head(x)
