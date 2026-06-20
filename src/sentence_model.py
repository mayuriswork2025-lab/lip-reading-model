import torch
import torch.nn as nn

from src.text_codec import NUM_CLASSES


class LipReadSentenceModel(nn.Module):
    """
    Sentence-level lip reading model: CNN frame encoder + BiGRU + CTC head.
    Inspired by LipNet but implemented in PyTorch for this project.
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256):
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
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.shape
        x = x.reshape(batch_size * timesteps, channels, height, width)
        x = self.cnn(x).reshape(batch_size, timesteps, -1)
        x, _ = self.gru(x)
        return self.fc(x)
