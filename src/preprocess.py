import os
from pathlib import Path

import cv2
import numpy as np

from src.face_detector import crop_mouth_sequence
from src.frame_extractor import extract_frames


def process_video(video_path, save_dir=None, delete_original=False, max_frames=75, output_size=(96, 96)):
    frames = extract_frames(video_path, max_frames=max_frames)
    mouth_crops = crop_mouth_sequence(frames, output_size=output_size)
    mouth_crops = [
        crop if crop is not None else np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        for crop in mouth_crops
    ]

    arr = np.stack(mouth_crops).astype(np.uint8)
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = Path(video_path).stem
        save_path = os.path.join(save_dir, f"{stem}.npy")
        with open(save_path, "wb") as handle:
            np.save(handle, arr, allow_pickle=False)

    if delete_original and os.path.exists(video_path):
        os.remove(video_path)

    return arr, save_path


def mouth_frames_to_jpegs(arr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for index, frame in enumerate(arr):
        path = os.path.join(output_dir, f"frame_{index:03d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths
