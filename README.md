# LipReader — Word-Level Lip Reading with CNN + LSTM

A from-scratch deep learning system that reads lips from short silent video clips and predicts which of 12 words was spoken. Built end-to-end: video preprocessing, a custom PyTorch CNN+LSTM model, a FastAPI backend, and a React frontend with live lip-region tracking.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Running Inference](#running-inference)
- [Running the Full App](#running-the-full-app)

## Overview

Given a short video clip of someone's mouth speaking a single word, the model predicts which word was said out of 12 possible classes. The pipeline:

```
raw video → MediaPipe lip detection → grayscale 96x96 crop, 20 frames
          → CNN (per-frame features) → LSTM (temporal sequence) → word prediction
```

The 12 target words are: **again, bin, blue, green, lay, now, place, please, red, set, soon, white** — a subset of the command vocabulary used in the GRID Corpus.

## Dataset

**Source:** [GRID Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/) — a publicly available audiovisual sentence corpus, originally created for speech perception research. Each sentence follows a fixed six-word grammar (command + color + preposition + letter + digit + adverb), recorded from 34 speakers.

For this project we extracted individual word-level clips for our 12-word vocabulary from a subset of GRID speakers.

- **Speakers used:** s1, s2, s14, s26
- **Clips per word per speaker:** 15
- **Expected total clips:** 720
- **Raw input:** `.mpg` video files from GRID, each aligned with a `.align` transcript marking word boundaries
- **Processed format:** each clip is preprocessed into a NumPy array of shape `(20, 96, 96)` — 20 grayscale frames of the cropped mouth region, saved as `.npy`

Processed data lives in `data/processed/<word>/clip_*.npy`, organized into one folder per word class. This directory is gitignored due to size — regenerate it locally from the raw GRID clips using the preprocessing scripts.

## Model Architecture

A two-stage CNN + LSTM design, implemented from scratch in PyTorch (`from_scratch/model.py`):

**Stage 1 — Per-frame CNN (`LipReaderCNN`)**
Extracts a compact feature vector from each individual frame:

| Layer | Details |
|---|---|
| Conv2d | 1 → 16 channels, 3×3 kernel, padding 1 |
| ReLU | — |
| MaxPool2d | 2×2 → halves spatial size |
| Conv2d | 16 → 32 channels, 3×3 kernel, padding 1 |
| ReLU | — |
| MaxPool2d | 2×2 → halves spatial size again |
| Flatten | 32 × 24 × 24 = 18,432 |
| Linear | 18,432 → 128 |

Input: one grayscale frame, 96×96. Output: a 128-dimensional feature vector summarizing that frame's mouth shape.

**Stage 2 — Temporal LSTM (`LipReaderModel`)**
The same CNN (shared weights) runs across all 20 frames of a clip, producing a sequence of 20 feature vectors. This sequence feeds into:

| Layer | Details |
|---|---|
| LSTM | input_size=128, hidden_size=128, 1 layer, batch_first |
| Linear (classifier) | 128 → 12 (one logit per word class) |

The LSTM's final hidden state — a single 128-number summary of the whole clip's temporal dynamics — is passed to a linear classifier producing 12 raw scores (logits), one per word. A softmax over these gives the final word probabilities.

**Input shape:** `(batch_size, 20, 1, 96, 96)` — batch of clips, 20 frames each, 1 grayscale channel, 96×96 pixels
**Output shape:** `(batch_size, 12)` — one score per word class

## Project Structure

```
lip-reading-model/
├── from_scratch/
│   ├── model.py            # CNN + LSTM architecture
│   ├── train.py             # training loop
│   ├── dataset.py           # PyTorch Dataset for loading processed clips
│   ├── dummy_data.py        # fake-data generator + shared constants (NUM_FRAMES, IMG_HEIGHT, etc.)
│   ├── inference.py          # predict_word() — loads model, runs inference
│   ├── labels.json          # word ↔ index mapping
│   └── checkpoints/
│       └── best_model.pth   # trained weights
├── backend/
│   ├── main.py               # FastAPI app — POST /predict endpoint
│   ├── preprocess_upload.py  # MediaPipe-based lip extraction for live uploads
│   └── demo_clips/           # 12 pre-recorded demo videos, one per word
├── frontend-new/              # React + Vite UI (retro pixel theme, live lip preview)
├── frontend/                  # original UI (kept for reference)
├── preprocess.py              # batch preprocessing script for building the training dataset
├── our_predict.py             # standalone CLI inference wrapper
└── requirements.txt
```

## Setup

**Backend (Python):**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend (Node):**

```bash
cd frontend-new
npm install
```

## Training

Training is handled by `from_scratch/train.py`.

```bash
cd from_scratch
python train.py
```

Key settings (edit at the top of `train.py`):

| Setting | Value | Notes |
|---|---|---|
| `USE_DUMMY_DATA` | `True` / `False` | `True` trains on random fake data to sanity-check the pipeline; set `False` to train on real GRID clips |
| `DATA_DIR` | `../data/processed` | expects one subfolder per word, each containing `.npy` clips |
| `BATCH_SIZE` | `4` | safe default for a free Colab T4 GPU; increase if you have more memory |
| `NUM_EPOCHS` | `20` | starting point — tune based on validation performance |
| `LEARNING_RATE` | `1e-3` | standard Adam optimizer starting point |

The trained weights are saved to `from_scratch/checkpoints/best_model.pth`.

To train on real data instead of dummy data, set `USE_DUMMY_DATA = False` and make sure `data/processed/<word>/*.npy` exists for all 12 words (generate this with `preprocess.py` from raw GRID `.mpg` clips first).

## Running Inference

**Quick CLI test** on a single video file:

```bash
python our_predict.py path/to/video.mp4
```

This runs the full pipeline (lip extraction → model → prediction) and prints the predicted word.

**Programmatic use** (e.g. inside a script or notebook):

```python
from from_scratch.inference import predict_word
from backend.preprocess_upload import preprocess_upload_video

clip = preprocess_upload_video("my_video.mp4")   # → numpy array, shape (20, 96, 96)
result = predict_word(clip)
print(result)
# {'predicted_word': 'bin', 'confidence': 0.94, 'all_scores': {...}}
```

`predict_word()` returns a dictionary with the top predicted word, its confidence score, and the full score distribution across all 12 words.

## Running the Full App

**1. Start the backend:**

```bash
uvicorn backend.main:app --reload
```
Runs at `http://127.0.0.1:8000`. Health check: `GET /health`. Prediction endpoint: `POST /predict` (expects a `video` file in form-data, returns `{predicted_word, confidence, all_scores}`).

**2. Start the frontend:**

```bash
cd frontend-new
npm run dev
```
Opens at `http://localhost:5173` (or similar — check terminal output).

**3. Try it out:**
Upload one of the pre-recorded clips in `backend/demo_clips/` (or record your own), and the app will show the live lip-region preview alongside the predicted word and per-word confidence scores.

---

Built by the TAM-VIT for a from-scratch lip reading project, migrating away from an earlier LipNet-based approach to this custom CNN+LSTM model.