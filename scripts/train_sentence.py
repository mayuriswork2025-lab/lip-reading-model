import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sentence_dataset import SentenceDataset
from src.sentence_model import LipReadSentenceModel
from src.text_codec import BLANK

DEFAULT_DATA = ROOT / "data" / "sentence_crops"
DEFAULT_MODEL = ROOT / "models" / "sentence_reader.pt"


def discover_sentence_samples(data_dir):
    root = Path(data_dir)
    samples = []
    for npy_path in sorted(root.glob("*.npy")):
        txt_path = npy_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        samples.append((str(npy_path), text))
    return samples


def collate_batch(batch):
    frames = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    label_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    input_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    targets = torch.cat(labels)
    return frames, targets, input_lengths, label_lengths


def train_sentence_model(
    data_dir=DEFAULT_DATA,
    output_path=DEFAULT_MODEL,
    epochs=25,
    batch_size=4,
    learning_rate=1e-3,
):
    samples = discover_sentence_samples(data_dir)
    if len(samples) < 2:
        raise RuntimeError(
            f"Need sentence samples in {data_dir}. Run scripts/bootstrap_sentence.py first."
        )

    loader = DataLoader(
        SentenceDataset(samples),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LipReadSentenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for frames, targets, input_lengths, label_lengths in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(frames)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)

            optimizer.zero_grad()
            loss = criterion(log_probs, targets, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"epoch {epoch}/{epochs} ctc_loss={running_loss / max(len(loader), 1):.4f}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)
    print(f"Saved sentence model to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train_sentence_model(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
