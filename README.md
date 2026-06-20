# LipRead Studio — Lip Reading (Word + Sentence)

A from-scratch **CNN + BiGRU** lip-reading system with **word classification** and **sentence-level CTC** decoding. Includes mouth-region extraction (MediaPipe), training pipelines, and a web demo.

---

## Two models (both built from scratch)

| Model | Task | File | Inference |
|-------|------|------|-----------|
| **Word classifier** | Predict one word (6 classes) | `models/word_classifier.pt` | `word_predict.py` |
| **Sentence reader (CTC)** | Predict full sentence (characters) | `models/sentence_reader.pt` | `sentence_predict.py` |

**Default demo uses the sentence reader.** Falls back to word-segment stitching if CTC model is missing.

---

## Deliverables checklist

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Dataset selected & preprocessing | Done | GRID sample clip + `data/mouth_crops/` |
| 2 | Lip region extraction from frames | Done | `src/face_detector.py`, `src/preprocess.py` |
| 3 | Sequence model + predictions | Done | `src/model.py`, `sentence_predict.py` |
| 4 | Demo short clip | Done | `evaluation/samples/id2_vcd_swwp2s.mpg` |
| 5 | README (this file) | Done | `README.md` |

---

## Demo short clip

**File:** `evaluation/samples/id2_vcd_swwp2s.mpg`

| Property | Value |
|----------|-------|
| Source | [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) — speaker s2 |
| Duration | ~3 seconds |
| Ground-truth sentence | `set white with p two soon` |
| Alignment file | `evaluation/samples/swwp2s.align` |

Use this clip to test inference locally or in the web UI.

---

## Dataset source

### Primary dataset: GRID Corpus (subset)

- **Official site:** http://spandh.dcs.shef.ac.uk/gridcorpus/
- **Paper reference:** Assael et al., *LipNet: End-to-End Sentence-level Lipreading* ([arXiv:1611.01599](https://arxiv.org/abs/1611.01599))
- **Why GRID:** Fixed ~51-word vocabulary, studio lighting, 25 fps, alignment files included

### What we use in this project

| Item | Details |
|------|---------|
| Raw video | 1 clip from GRID speaker s2 (`id2_vcd_swwp2s.mpg`) |
| Alignments | Word-level timestamps in `swwp2s.align` |
| Processed data | `data/mouth_crops/` — 127 `.npy` files (6 word classes) |
| Augmentation | Flip, brightness jitter, frame shift (20 copies per word) |

### Preprocessing pipeline

```
Raw video (.mpg)
    → Frame extraction (75 frames max, 25 fps)
    → Face detection + lip landmark crop (MediaPipe, 96×96 px)
    → NumPy array saved as .npy  [T × H × W × C]
```

**Scripts:**
- `scripts/bootstrap_demo.py` — build training set from sample clip
- `src/preprocess.py` — video → mouth crop sequence
- `src/face_detector.py` — MediaPipe lip landmarks + Haar fallback

### Expanding to more speakers (optional)

Download GRID speakers s1–s3 (~2–3 GB) and run:

```powershell
python scripts/bootstrap_demo.py   # single-clip demo
# For multi-speaker: use scripts/preprocess_grid.py (when GRID raw data is present)
```

---

## Model architecture

**Type:** Word-level sequence classifier (CNN spatial encoder + temporal GRU)

```
Input: mouth frame sequence  [B, T, 3, 96, 96]
              │
              ▼
    ┌─────────────────────┐
    │  CNN (per frame)    │  Conv2d 3→32→64→128 + MaxPool + AdaptiveAvgPool(4)
    │  Feature: 2048-d    │
    └─────────┬───────────┘
              │  reshape → [B, T, 2048]
              ▼
    ┌─────────────────────┐
    │  BiGRU (2 layers)   │  hidden=256, bidirectional → 512-d per step
    └─────────┬───────────┘
              │  temporal mean pooling
              ▼
    ┌─────────────────────┐
    │  Linear classifier  │  512 → num_classes
    └─────────┬───────────┘
              ▼
    Softmax → predicted word
```

| Layer | Details |
|-------|---------|
| CNN | 3× Conv2d blocks (32, 64, 128 channels), ReLU, MaxPool |
| Sequence | 2-layer Bidirectional GRU, 256 hidden units |
| Pooling | Mean over time steps |
| Output | Softmax over word classes |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3) |

**Implementation:** `src/model.py` (words), `src/sentence_model.py` (sentences + CTC)

### Sentence model (CTC)

```
Full mouth sequence → CNN → BiGRU → per-frame char logits → CTC decode → sentence text
```

| Item | Value |
|------|-------|
| Loss | CTC (Connectionist Temporal Classification) |
| Characters | a–z + space + blank (28 classes) |
| Training data | `data/sentence_crops/` (31 full-sentence clips) |
| Train script | `scripts/train_sentence.py` |
| Demo output | `set wi on` (partial; ground truth: `set white with p two soon`) |

---

## Model architecture (word classifier)

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch 2.2 |
| Epochs | 8–12 |
| Batch size | 4–8 |
| Max frames per clip | 40 |
| Train/val split | 80/20 stratified by label |
| Hardware | CPU (local) or GPU (Google Colab) |
| Output | `models/word_classifier.pt` |

### Train sentence reader (CTC)

```powershell
python scripts/bootstrap_sentence.py
python scripts/train_sentence.py --epochs 20
```

### Train word classifier

```powershell
cd D:\lip-reading-model
$env:PYTHONPATH = "D:\lip-reading-model"

# Step 1: Build mouth-crop dataset from demo clip
.\.venv\Scripts\python.exe scripts\bootstrap_demo.py

# Step 2: Train CNN+GRU model
.\.venv\Scripts\python.exe scripts\train_words.py --epochs 12 --batch-size 4
```

### Train on GPU (Colab)

Open `notebooks/colab_train.ipynb`, enable GPU runtime, run all cells, download `word_classifier.pt`.

### Checkpoint contents

```python
{
    "state_dict": ...,      # model weights
    "labels": [...],        # word list
    "label_to_idx": {...}   # label mapping
}
```

---

## How to run inference

### Option 1 — Sentence inference (recommended)

```powershell
python scripts/run_inference.py --mode sentence
```

### Option 2 — Word inference

```powershell
python scripts/run_inference.py --mode word
```

### Option 3 — Command line (legacy)

```powershell
cd D:\lip-reading-model
$env:PYTHONPATH = "D:\lip-reading-model"

# Demo clip
.\.venv\Scripts\python.exe scripts\run_inference.py

# Your own video
.\.venv\Scripts\python.exe scripts\run_inference.py path\to\your_video.mp4
```

**Example output:**
```
Prediction:  white
Confidence:  17.9%
Top predictions:
  1. white     17.9%
  2. set       16.2%
  ...
```

### Option 2 — Python API

```python
from word_predict import predict_from_video

word, confidence = predict_from_video("evaluation/samples/id2_vcd_swwp2s.mpg")
print(word, confidence)
```

### Option 3 — Web UI

```powershell
.\start.ps1 -OpenBrowser
```

1. Open http://127.0.0.1:5173
2. Upload `evaluation\samples\id2_vcd_swwp2s.mpg`
3. Click **Run LipRead Studio**
4. View mouth frame gallery + prediction

### Option 4 — FastAPI

```powershell
# Health check
curl http://127.0.0.1:8000/health

# Upload + predict (via UI or POST /upload then GET /predict/{job_id})
```

### Quick sanity test

```powershell
.\.venv\Scripts\python.exe scripts\test_demo.py
```

---

## Project structure

```
D:\lip-reading-model\
├── README.md                       # This file
├── models\word_classifier.pt       # Trained sequence model
├── data\mouth_crops\               # Preprocessed training clips
├── evaluation\samples\
│   ├── id2_vcd_swwp2s.mpg          # Demo short clip
│   └── swwp2s.align                # Word alignments
├── src\
│   ├── model.py                    # CNN + BiGRU architecture
│   ├── face_detector.py            # Lip region extraction
│   ├── preprocess.py               # Video → mouth crops
│   ├── dataset.py                  # PyTorch Dataset
│   └── data_loader.py              # Alignment parsing, splits
├── scripts\
│   ├── bootstrap_demo.py           # Build training data
│   ├── train_words.py              # Train model
│   ├── run_inference.py            # Run predictions
│   └── test_demo.py                # End-to-end test
├── backend\                        # FastAPI server
├── frontend\                       # React upload UI
├── setup.ps1                       # First-time setup
└── start.ps1                       # Launch demo
```

---

## Setup (first time)

```powershell
cd D:\lip-reading-model
.\setup.ps1
```

Requires **Python 3.10+** and **Node.js 18+**.

---

## Team roles

| Role | Files |
|------|-------|
| ML Engineer | `src/model.py`, `scripts/train_words.py` |
| Data Engineer | `scripts/bootstrap_demo.py`, `data/mouth_crops/` |
| Backend Developer | `backend/`, `src/preprocess.py` |
| Frontend Developer | `frontend/` |
| Project Lead | `scripts/test_demo.py`, `scripts/run_inference.py` |

---

## References

- GRID Corpus: http://spandh.dcs.shef.ac.uk/gridcorpus/
- LipNet paper: https://arxiv.org/abs/1611.01599
- MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh.html
- Original repo base: https://github.com/tanvih23/lip-reading-model

---

## License

MIT License (see `LICENSE`)
