"""
Build a tiny training set from the bundled GRID sample clip + align file.
No large dataset download required. Augmentation expands 6 words into ~120 clips.
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
DEFAULT_OUTPUT = ROOT / "data" / "mouth_crops"


def extract_word_clip(video_path, start_frame, end_frame, max_frames=75):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= index < end_frame:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        index += 1
    cap.release()
    return frames


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
    if rng.random() < 0.4:
        shift = int(rng.integers(-2, 3))
        if shift:
            augmented = augmented[shift:] + augmented[:shift]
    return augmented


def save_word_clip(frames, label, output_dir, clip_name):
    from src.face_detector import crop_mouth

    mouth_crops = []
    for frame in frames:
        crop = crop_mouth(frame, output_size=(96, 96))
        if crop is None:
            crop = np.zeros((96, 96, 3), dtype=np.uint8)
        mouth_crops.append(crop)

    arr = np.stack(mouth_crops).astype(np.uint8)
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    save_path = label_dir / f"{clip_name}.npy"
    np.save(save_path, arr)
    return save_path


def bootstrap(video_path=DEFAULT_VIDEO, align_path=DEFAULT_ALIGN, output_dir=DEFAULT_OUTPUT, copies_per_word=20):
    if not video_path.exists():
        raise FileNotFoundError(
            f"Sample video missing at {video_path}. Place a front-facing clip there or pass --video."
        )
    if not align_path.exists():
        raise FileNotFoundError(f"Alignment file missing at {align_path}")

    words = parse_alignment(align_path)
    if not words:
        raise RuntimeError("No words found in alignment file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for word_index, (start, end, word) in enumerate(words):
        base_frames = extract_word_clip(video_path, start, end)
        if not base_frames:
            continue
        save_word_clip(base_frames, word, output_dir, f"{word}_base")
        saved.append(output_dir / word / f"{word}_base.npy")
        for copy_index in range(copies_per_word):
            aug = augment_frames(base_frames, seed=word_index * 1000 + copy_index)
            path = save_word_clip(aug, word, output_dir, f"{word}_{copy_index:02d}")
            saved.append(path)

    full_arr, _ = process_video(str(video_path), save_dir=str(output_dir / "_full"))
    print(f"Bootstrap complete: {len(saved)} word clips across {len(words)} labels")
    if full_arr is not None:
        print(f"Full-video mouth crop saved to {output_dir / '_full'}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--align", default=str(DEFAULT_ALIGN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copies-per-word", type=int, default=20)
    args = parser.parse_args()
    bootstrap(
        video_path=Path(args.video),
        align_path=Path(args.align),
        output_dir=Path(args.output),
        copies_per_word=args.copies_per_word,
    )
