import torch
import torch.nn as nn


class LipReadModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.gru = nn.GRU(
            128 * 4 * 4,
            256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.shape
        x = x.reshape(batch_size * timesteps, channels, height, width)
        x = self.cnn(x).reshape(batch_size, timesteps, -1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.fc(x)
