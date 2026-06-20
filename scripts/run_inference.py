"""
Run inference on a video clip using the trained CNN+GRU sequence model.

Example:
  python scripts/run_inference.py evaluation/samples/id2_vcd_swwp2s.mpg
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DEMO = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"


def main():
    parser = argparse.ArgumentParser(description="Lip-reading word inference")
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_DEMO),
        help="Path to input video (.mpg, .mp4, .mov)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    from word_predict import predict_from_video, predict_top_k, _load_model
    from src.preprocess import process_video

    print(f"Video:  {video_path}")
    print(f"Model:  {ROOT / 'models' / 'word_classifier.pt'}")
    print()

    _, _, labels = _load_model()
    print(f"Vocabulary ({len(labels)} words): {', '.join(labels)}")
    print("Processing...")

    mouth_arr, _ = process_video(str(video_path), max_frames=75, output_size=(96, 96))
    word, confidence = predict_from_video(str(video_path))
    top_k = predict_top_k(mouth_arr, k=args.top_k)

    print()
    print(f"Prediction:  {word}")
    print(f"Confidence:  {confidence * 100:.1f}%")
    print(f"Frames used: {len(mouth_arr)}")
    print()
    print("Top predictions:")
    for rank, (label, prob) in enumerate(top_k, start=1):
        print(f"  {rank}. {label:8s}  {prob * 100:.1f}%")

    if video_path == DEFAULT_DEMO:
        print()
        print("Demo clip ground truth (GRID alignment): set white with p two soon")
        print("(Model predicts the dominant word in the full clip.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
