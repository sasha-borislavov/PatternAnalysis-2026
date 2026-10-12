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
