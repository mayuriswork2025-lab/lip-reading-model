# LipRead Studio — Team Project

Lip-reading **word classifier** built from scratch: **CNN + GRU** (PyTorch), mouth-crop pipeline (MediaPipe), FastAPI backend, React frontend.

**No large GRID download required.** Trained on the bundled sample video (6 words, 127 augmented clips).

## Team roles

| Role | Folder | Task |
|------|--------|------|
| ML Engineer | `src/model.py`, `scripts/train_words.py` | Model + training |
| Data Engineer | `scripts/bootstrap_demo.py`, `data/mouth_crops/` | Dataset prep |
| Backend Developer | `backend/`, `src/preprocess.py` | Video → mouth crops → API |
| Frontend Developer | `frontend/` | Upload UI + frame gallery |
| Project Lead | `scripts/test_demo.py` | End-to-end testing |

## Prerequisites

- **Python 3.10+** — https://www.python.org/downloads/
- **Node.js 18+** — https://nodejs.org/
- **Windows** (PowerShell) or adapt commands for Mac/Linux

## First-time setup (each teammate)

### Option A — PowerShell (recommended)

```powershell
cd D:\lip-reading-model
.\setup.ps1
```

### Option B — Command Prompt

```cmd
cd D:\lip-reading-model
setup.bat
```

This installs Python deps, npm packages, and verifies the trained model.

## Run the demo

```powershell
cd D:\lip-reading-model
.\start.ps1 -OpenBrowser
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:5173 | Web UI — upload video, see results |
| http://127.0.0.1:8000/health | Backend health check |

**Test video:** `evaluation\samples\id2_vcd_swwp2s.mpg`

## Quick test (no browser)

```powershell
cd D:\lip-reading-model
$env:PYTHONPATH = "D:\lip-reading-model"
.\.venv\Scripts\python.exe scripts\test_demo.py
```

## Trained vocabulary

`set` · `white` · `with` · `p` · `two` · `soon`

## Project structure

```
D:\lip-reading-model\
├── models\word_classifier.pt     # Trained model (included)
├── data\mouth_crops\             # Training data (included)
├── evaluation\samples\             # Test video
├── src\                           # Preprocessing + CNN+GRU
├── backend\                       # FastAPI job API
├── frontend\                      # React UI
├── scripts\                       # Bootstrap, train, test
├── notebooks\colab_train.ipynb    # Optional GPU retrain on Colab
├── setup.ps1 / setup.bat          # First-time setup
└── start.ps1                      # Launch demo
```

## VS Code

1. **File → Open Folder** → `D:\lip-reading-model`
2. Terminal → run `.\setup.ps1` once
3. Terminal → run `.\start.ps1 -OpenBrowser`

## Retrain (optional)

```powershell
$env:PYTHONPATH = "D:\lip-reading-model"
.\.venv\Scripts\python.exe scripts\bootstrap_demo.py
.\.venv\Scripts\python.exe scripts\train_words.py --epochs 8
```

For faster training use `notebooks\colab_train.ipynb` on Google Colab GPU.

## Share this folder

**Option 1 — Zip (recommended for teammates on other PCs)**

```powershell
cd D:\lip-reading-model
.\package_for_team.ps1
```

Creates `D:\lip-reading-model-team.zip` (~170 MB). Teammates unzip anywhere, then run `setup.ps1`.

**Option 2 — Copy folder directly** (same PC or network drive)

Copy `D:\lip-reading-model` to a shared location. If teammates are on the **same machine**, the included `.venv` may work as-is. On a **new PC**, each person runs `setup.ps1`.

**Do not include in zip:** `.venv`, `node_modules`, `data\jobs` (recreated by setup).

