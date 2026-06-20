"""
Build sentence-level training samples from the demo GRID clip.
Each sample = full mouth-frame sequence + full sentence text.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.data_loader import parse_alignment
from src.preprocess import process_video

DEFAULT_VIDEO = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"
DEFAULT_ALIGN = ROOT / "evaluation" / "samples" / "swwp2s.align"
DEFAULT_OUTPUT = ROOT / "data" / "sentence_crops"


def sentence_from_alignment(align_path):
    words = parse_alignment(align_path)
    return " ".join(word for _, _, word in words)


def augment_frames(frames, seed):
    rng = np.random.default_rng(seed)
    augmented = [frame.copy() for frame in frames]
    if rng.random() < 0.5:
        augmented = [cv2.flip(frame, 1) for frame in augmented]
    if rng.random() < 0.7:
        factor = float(rng.uniform(0.85, 1.15))
        augmented = [
            np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            for frame in augmented
        ]
    return augmented


def bootstrap_sentence(
    video_path=DEFAULT_VIDEO,
    align_path=DEFAULT_ALIGN,
    output_dir=DEFAULT_OUTPUT,
    copies=30,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video missing: {video_path}")
    if not align_path.exists():
        raise FileNotFoundError(f"Alignment missing: {align_path}")

    sentence = sentence_from_alignment(align_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    from src.face_detector import crop_mouth_sequence

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    base_crops = crop_mouth_sequence(frames, output_size=(96, 96))
    saved = []

    def save_clip(crop_frames, name):
        arr = np.stack(crop_frames).astype(np.uint8)
        path = output_dir / f"{name}.npy"
        np.save(path, arr)
        meta_path = output_dir / f"{name}.txt"
        meta_path.write_text(sentence, encoding="utf-8")
        saved.append((path, sentence))

    save_clip(base_crops, "demo_base")
    for index in range(copies):
        aug_frames = augment_frames(frames, seed=index)
        aug_crops = crop_mouth_sequence(aug_frames, output_size=(96, 96))
        save_clip(aug_crops, f"demo_{index:02d}")

    print(f"Sentence: {sentence}")
    print(f"Saved {len(saved)} sentence clips to {output_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--align", default=str(DEFAULT_ALIGN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copies", type=int, default=30)
    args = parser.parse_args()
    bootstrap_sentence(
        video_path=Path(args.video),
        align_path=Path(args.align),
        output_dir=Path(args.output),
        copies=args.copies,
    )
