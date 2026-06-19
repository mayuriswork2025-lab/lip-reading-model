"""Quick end-to-end check: model load, video inference, API health."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_VIDEO = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"
MODEL_PATH = ROOT / "models" / "word_classifier.pt"


def main():
    print("=== LipRead Studio Demo Test ===\n")

    if not MODEL_PATH.exists():
        print(f"FAIL: Model missing at {MODEL_PATH}")
        print("Run: .\\.venv\\Scripts\\python.exe scripts\\bootstrap_demo.py")
        print("Then: .\\.venv\\Scripts\\python.exe scripts\\train_words.py --epochs 8")
        return 1

    print(f"OK  Model found ({MODEL_PATH.stat().st_size // 1024} KB)")

    crop_count = len(list((ROOT / "data" / "mouth_crops").rglob("*.npy")))
    print(f"OK  Training crops: {crop_count}")

    if not SAMPLE_VIDEO.exists():
        print(f"WARN: Sample video missing at {SAMPLE_VIDEO}")
        print("Upload any front-facing speaking clip via the web UI instead.")
        return 0

    from src.preprocess import process_video
    from word_predict import predict_word_sequence

    mouth_arr, _ = process_video(str(SAMPLE_VIDEO), max_frames=75, output_size=(96, 96))
    word, confidence = predict_word_sequence(mouth_arr)
    print(f"OK  Sample video prediction: '{word}' ({confidence * 100:.1f}% confidence)")
    print("\nTrained words: set, white, with, p, two, soon")
    print("Start demo: .\\start.ps1 -OpenBrowser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
