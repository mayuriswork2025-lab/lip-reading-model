from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.model import LipReadModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "word_classifier.pt"
MAX_FRAMES = 40


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run scripts/bootstrap_demo.py then scripts/train_words.py."
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    labels = checkpoint["labels"]
    label_to_idx = checkpoint["label_to_idx"]
    model = LipReadModel(num_classes=len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    idx_to_label = {index: label for label, index in label_to_idx.items()}
    return model, idx_to_label, labels


def _prepare_tensor(mouth_arr, max_frames=MAX_FRAMES):
    frames = mouth_arr.astype(np.float32) / 255.0
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)

    if len(frames) > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    elif len(frames) < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - len(frames), *frames.shape[1:]), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    return torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)


def predict_word_sequence(mouth_arr):
    """Run the CNN+GRU sequence model on mouth-crop frames."""
    model, idx_to_label, _ = _load_model()
    tensor = _prepare_tensor(mouth_arr)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)
        label = idx_to_label[int(pred_idx)]
    return label, float(confidence.item())


def predict_from_video(video_path):
    """Full pipeline: video -> mouth crops -> word prediction."""
    from src.preprocess import process_video

    mouth_arr, _ = process_video(video_path, max_frames=75, output_size=(96, 96))
    word, confidence = predict_word_sequence(mouth_arr)
    return word, confidence


def predict_top_k(mouth_arr, k=3):
    """Return top-k word predictions with probabilities."""
    model, idx_to_label, labels = _load_model()
    tensor = _prepare_tensor(mouth_arr)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, min(k, len(labels)))
    return [
        (idx_to_label[int(index)], float(prob.item()))
        for index, prob in zip(top_idx, top_probs)
    ]
