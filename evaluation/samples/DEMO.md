# Demo Short Clip

**File:** `evaluation/samples/id2_vcd_swwp2s.mpg`

| Field | Value |
|-------|-------|
| Dataset | [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) |
| Speaker | s2 |
| Sentence | set white with p two soon |
| Alignment | `evaluation/samples/swwp2s.align` |

## Run inference on this clip

```powershell
cd D:\lip-reading-model
$env:PYTHONPATH = "D:\lip-reading-model"
.\.venv\Scripts\python.exe scripts\run_inference.py
```

Or upload it in the web UI after running `.\start.ps1 -OpenBrowser`.

See **README.md** for full documentation.
