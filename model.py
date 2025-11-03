import torch
import torch.nn as nn

class DrumCNNv2(nn.Module):
    def __init__(self, n_mels, n_classes):
        super().__init__()

        # (B, 1, n_mels, time)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),     # downsample both freq & time

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # Global pooling over time & frequency -> (B, 128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = self.features(x)             # (B, 128, F', T')
        x = self.global_pool(x)          # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)
        x = self.classifier(x)           # (B, n_classes)
        return x
