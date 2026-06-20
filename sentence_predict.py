from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.sentence_model import LipReadSentenceModel
from src.text_codec import ctc_greedy_decode

ROOT = Path(__file__).resolve().parent
SENTENCE_MODEL_PATH = ROOT / "models" / "sentence_reader.pt"
MAX_FRAMES = 75


def _prepare_tensor(mouth_arr, max_frames=MAX_FRAMES):
    frames = mouth_arr.astype(np.float32) / 255.0
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)

    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        pad = np.zeros((max_frames - len(frames), *frames.shape[1:]), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    return torch.tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)


@lru_cache(maxsize=1)
def _load_sentence_model():
    if not SENTENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sentence model not found at {SENTENCE_MODEL_PATH}. "
            "Run: scripts/bootstrap_sentence.py then scripts/train_sentence.py"
        )
    checkpoint = torch.load(SENTENCE_MODEL_PATH, map_location="cpu")
    model = LipReadSentenceModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_sentence_ctc(mouth_arr):
    model = _load_sentence_model()
    tensor = _prepare_tensor(mouth_arr)
    with torch.no_grad():
        logits = model(tensor)
        log_probs = logits.log_softmax(dim=2)[0]
        sentence = ctc_greedy_decode(log_probs)
        probs = torch.softmax(logits, dim=2)[0]
        confidence = float(probs.max(dim=1).values.mean().item())
    return sentence.strip(), confidence


def predict_sentence_from_video(video_path):
    from src.preprocess import process_video

    mouth_arr, _ = process_video(video_path, max_frames=MAX_FRAMES, output_size=(96, 96))
    sentence, confidence = predict_sentence_ctc(mouth_arr)
    return sentence, confidence, "ctc"
