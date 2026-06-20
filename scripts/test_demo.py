"""Quick end-to-end check: sentence model load + inference."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_VIDEO = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"
MODEL_PATH = ROOT / "models" / "sentence_reader.pt"


def main():
    print("=== LipRead Sentence Model Test ===\n")

    if not MODEL_PATH.exists():
        print(f"FAIL: Model missing at {MODEL_PATH}")
        print("Run: scripts\\bootstrap_sentence.py")
        print("Then: scripts\\train_sentence.py --epochs 20")
        return 1

    print(f"OK  Sentence model found ({MODEL_PATH.stat().st_size // 1024} KB)")

    clip_count = len(list((ROOT / "data" / "sentence_crops").glob("*.npy")))
    print(f"OK  Sentence training clips: {clip_count}")

    if not SAMPLE_VIDEO.exists():
        print(f"WARN: Sample video missing at {SAMPLE_VIDEO}")
        return 0

    from sentence_predict import predict_sentence_from_video

    sentence, confidence, method = predict_sentence_from_video(str(SAMPLE_VIDEO))
    print(f"OK  Prediction: '{sentence}' ({confidence * 100:.1f}% confidence, {method})")
    print("\nGround truth: set white with p two soon")
    print("Start demo: .\\start.ps1 -OpenBrowser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
