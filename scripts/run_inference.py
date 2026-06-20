"""
Run sentence-level lip reading inference.

Example:
  python scripts/run_inference.py
  python scripts/run_inference.py evaluation/samples/id2_vcd_swwp2s.mpg
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DEMO = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"


def main():
    parser = argparse.ArgumentParser(description="Sentence lip-reading inference")
    parser.add_argument("video", nargs="?", default=str(DEFAULT_DEMO))
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    from sentence_predict import predict_sentence_from_video

    print(f"Video: {video_path}")
    sentence, confidence, method = predict_sentence_from_video(str(video_path))
    print(f"Sentence:   {sentence}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print(f"Method:     {method}")

    if video_path == DEFAULT_DEMO:
        print("\nGround truth: set white with p two soon")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
