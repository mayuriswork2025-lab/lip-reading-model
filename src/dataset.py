import numpy as np
import torch
from torch.utils.data import Dataset


class LipDataset(Dataset):
    def __init__(self, samples, max_frames=40):
        self.samples = samples
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path).astype(np.float32) / 255.0
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)

        if len(frames) > self.max_frames:
            frames = frames[: self.max_frames]
        elif len(frames) < self.max_frames:
            pad = np.zeros(
                (self.max_frames - len(frames), *frames.shape[1:]),
                dtype=np.float32,
            )
            frames = np.concatenate([frames, pad], axis=0)

        frames = torch.tensor(frames).permute(0, 3, 1, 2)
        return frames, label
