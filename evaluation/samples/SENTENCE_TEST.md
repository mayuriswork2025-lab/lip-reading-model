# Sentence test video

## Use this for sentence prediction

| File | Purpose |
|------|---------|
| **`sentence_demo.mp4`** | **Best for testing** — trimmed clip with full sentence only |
| `id2_vcd_swwp2s.mpg` | Full GRID sample (also works) |

**Ground truth sentence:** `set white with p two soon`

## Create the sentence demo clip

```powershell
cd D:\lip-reading-model
$env:PYTHONPATH = "D:\lip-reading-model"
.\.venv\Scripts\python.exe scripts\create_sentence_demo.py
```

## Run sentence prediction

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py evaluation\samples\sentence_demo.mp4
```

Expected (CTC model): partial sentence like `set wi on` — not a single word like `white`.

## If you only see `white`

You are **not** using the sentence model. Check:

1. Run: `scripts\run_inference.py` (not old word scripts)
2. Backend health: http://127.0.0.1:8000/health → `"model_type": "sentence_ctc"`
3. Model file exists: `models\sentence_reader.pt`

**Single word `white`** = word classifier on a short clip.  
**Sentence text** = sentence CTC model on `sentence_demo.mp4`.

## Download original GRID sample

https://github.com/rizkiarm/LipNet/raw/master/evaluation/samples/id2_vcd_swwp2s.mpg

Save to: `evaluation\samples\id2_vcd_swwp2s.mpg`
