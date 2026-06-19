# LipRead Studio — Quick Start

## Setup (first time only)

```powershell
cd D:\lip-reading-model
.\setup.ps1
```

## Run

```powershell
.\start.ps1 -OpenBrowser
```

- **UI:** http://127.0.0.1:5173
- **API:** http://127.0.0.1:8000/health
- **Test video:** `evaluation\samples\id2_vcd_swwp2s.mpg`

## Quick test

```powershell
$env:PYTHONPATH = "D:\lip-reading-model"
.\.venv\Scripts\python.exe scripts\test_demo.py
```

See **TEAM_README.md** for full team guide.
