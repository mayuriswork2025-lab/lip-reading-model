"""
Preprocess a raw uploaded video for inference (Member D backend).

Same lip pipeline as training data:
  raw video -> MediaPipe FaceLandmarker lip crop -> grayscale 96x96 -> (20, 96, 96) numpy array

Uses the new MediaPipe Tasks API (FaceLandmarker) since mediapipe>=0.10.30
removed the legacy mp.solutions.face_mesh API this code originally used.

Usage:
    from preprocess_upload import preprocess_upload_video

    clip = preprocess_upload_video("uploaded.mp4")   # shape (20, 96, 96)
    word = predict_word(clip)
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Must match extract_lips.py / training data
LIP_WIDTH = 96
LIP_HEIGHT = 96
PADDING = 0.20
TARGET_FRAMES = 20

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37,
    0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81,
    42, 183, 78,
]

# Where to cache the FaceLandmarker model bundle on disk
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Module-level singleton so we don't reload the model on every request
_landmarker = None


def _ensure_model_downloaded():
    """Download the FaceLandmarker .task bundle once, cache it on disk."""
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _ensure_model_downloaded()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def lip_bbox_from_landmarks(landmarks, width, height, padding=PADDING):
    xs = [landmarks[i].x * width for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * height for i in LIP_LANDMARKS]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_and_resize(frame_gray, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (LIP_WIDTH, LIP_HEIGHT), interpolation=cv2.INTER_AREA)


def normalize_frame_count(frames, target=TARGET_FRAMES):
    t = len(frames)
    if t == 0:
        raise ValueError("No frames to normalize")
    if t == target:
        return frames
    if t > target:
        idx = np.linspace(0, t - 1, target).astype(int)
        return [frames[i] for i in idx]
    out = list(frames)
    last = frames[-1]
    while len(out) < target:
        out.append(last.copy())
    return out


def preprocess_upload_video(video_path, start_frame=None, end_frame=None):
    """
    Process a raw uploaded video into a training-compatible numpy clip.

    Args:
        video_path: path to .mp4, .mpg, .avi, etc.
        start_frame: optional first frame (default: 0)
        end_frame: optional last frame (default: end of video)

    Returns:
        numpy array shape (20, 96, 96), dtype uint8, grayscale lip crops

    Raises:
        FileNotFoundError, RuntimeError if video cannot be read or no face found
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = 0 if start_frame is None else max(0, int(start_frame))
    end = (total - 1) if end_frame is None else min(int(end_frame), total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    landmarker = _get_landmarker()

    frames = []
    last_bbox = None
    # IMPORTANT: the landmarker instance is a module-level singleton reused
    # across requests. MediaPipe's VIDEO mode tracks the last timestamp it
    # saw *per landmarker instance*, not per video — so starting back at 0
    # for every new request will be "less than" the previous request's final
    # timestamp and trigger "Input timestamp must be monotonically increasing."
    # Using a monotonic wall-clock-based timestamp guarantees it always moves
    # forward, no matter how many videos this same landmarker has processed.
    timestamp_ms = int(time.monotonic() * 1000)
    # Guard against fps metadata being 0, very high, or otherwise unreliable —
    # always advance by at least 1ms so timestamps are strictly increasing.
    frame_duration_ms = max(1, int(1000 / fps)) if fps > 0 else 33

    for _ in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly increasing for VIDEO running mode
        timestamp_ms += frame_duration_ms
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        bbox = None
        if result.face_landmarks:
            bbox = lip_bbox_from_landmarks(result.face_landmarks[0], w, h)
        if bbox is None and last_bbox is not None:
            bbox = last_bbox
        if bbox is None:
            continue

        last_bbox = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lip = crop_and_resize(gray, bbox)
        if lip is not None:
            frames.append(lip)

    cap.release()

    if not frames:
        raise RuntimeError("No lip frames extracted — face not detected in video")

    frames = normalize_frame_count(frames)
    return np.stack(frames, axis=0).astype(np.uint8)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_upload.py <video_path>")
        sys.exit(1)
    arr = preprocess_upload_video(sys.argv[1])
    print("OK shape:", arr.shape, "dtype:", arr.dtype)