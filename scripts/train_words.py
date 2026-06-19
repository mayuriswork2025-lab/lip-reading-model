import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import discover_word_samples, train_val_split
from src.dataset import LipDataset
from src.model import LipReadModel

DEFAULT_DATA = ROOT / "data" / "mouth_crops"
DEFAULT_MODEL = ROOT / "models" / "word_classifier.pt"


def collate_fn(batch, label_to_idx):
    frames = torch.stack([item[0] for item in batch])
    labels = torch.tensor([label_to_idx[item[1]] for item in batch], dtype=torch.long)
    return frames, labels


def train_model(
    data_dir=DEFAULT_DATA,
    output_path=DEFAULT_MODEL,
    epochs=15,
    batch_size=4,
    learning_rate=1e-3,
):
    samples = discover_word_samples(data_dir)
    if len(samples) < 4:
        raise RuntimeError(
            f"Need at least 4 mouth-crop samples in {data_dir}. Run scripts/bootstrap_demo.py first."
        )

    labels = sorted({label for _, label in samples})
    label_to_idx = {label: index for index, label in enumerate(labels)}
    train_samples, val_samples = train_val_split(samples)

    train_loader = DataLoader(
        LipDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, label_to_idx),
    )
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            LipDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, label_to_idx),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LipReadModel(num_classes=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for frames, targets in train_loader:
            frames = frames.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device) if val_loader else None
        train_acc = evaluate(model, train_loader, device)
        message = (
            f"epoch {epoch}/{epochs} loss={running_loss / max(len(train_loader), 1):.4f} "
            f"train_acc={train_acc:.2f}"
        )
        if val_acc is not None:
            message += f" val_acc={val_acc:.2f}"
            if val_acc >= best_acc:
                best_acc = val_acc
                save_checkpoint(output_path, model, labels, label_to_idx)
        else:
            save_checkpoint(output_path, model, labels, label_to_idx)
        print(message, flush=True)

    if not output_path.exists():
        save_checkpoint(output_path, model, labels, label_to_idx)

    print(f"Saved model to {output_path}")
    return output_path


def evaluate(model, loader, device):
    if loader is None:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            logits = model(frames)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / max(total, 1)


def save_checkpoint(path, model, labels, label_to_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "label_to_idx": label_to_idx,
        },
        path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train_model(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
