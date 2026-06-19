# Model Weights

This file lists the LipNet model weight files and where to download them. Do NOT commit the actual .h5 files into the repo; host them on a shared file store (S3, Google Drive, internal share).

Files expected in `evaluation/models/`:

- `unseen-weights178.h5`
- `overlapped-weights368.h5`

Example placeholders (replace with your hosting links and checksums):

| Filename | URL | SHA256 |
|---|---|---|
| unseen-weights178.h5 | https://example.com/unseen-weights178.h5 | <sha256-here>
| overlapped-weights368.h5 | https://example.com/overlapped-weights368.h5 | <sha256-here>

Download instructions

Windows PowerShell:

```powershell
.\download_weights.ps1 -url "https://example.com/overlapped-weights368.h5"
```

Unix:

```bash
./download_weights.sh https://example.com/overlapped-weights368.h5
```

Verification (optional): compute SHA256 locally and compare to the value in this file.

```bash
sha256sum evaluation/models/overlapped-weights368.h5
```

Replace the example URLs with the real hosted URLs and the correct SHA256 checksums before sharing the repo.
