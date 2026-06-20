"""
Create a trimmed sentence demo video from the GRID sample + alignment.
Use this clip to test sentence-level prediction (not single-word clips).

Output: evaluation/samples/sentence_demo.mp4
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.data_loader import parse_alignment

SOURCE = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"
ALIGN = ROOT / "evaluation" / "samples" / "swwp2s.align"
OUTPUT = ROOT / "evaluation" / "samples" / "sentence_demo.mp4"


def extract_sentence_clip(video_path, align_path, output_path):
    words = parse_alignment(align_path)
    if not words:
        raise RuntimeError("No words in alignment file")

    start_frame = words[0][0]
    end_frame = words[-1][1]
    sentence = " ".join(w for _, _, w in words)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    index = 0
    written = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= index < end_frame:
            out.write(frame)
            written += 1
        index += 1
        if index >= end_frame:
            break

    cap.release()
    out.release()

    print(f"Sentence: {sentence}")
    print(f"Frames: {start_frame}-{end_frame} ({written} frames)")
    print(f"Saved: {output_path}")
    return sentence, output_path


if __name__ == "__main__":
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Download from:\n"
            "https://github.com/rizkiarm/LipNet/raw/master/evaluation/samples/id2_vcd_swwp2s.mpg"
        )
    extract_sentence_clip(SOURCE, ALIGN, OUTPUT)
