import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.jobs import JOBS_DIR, create_job, get_job

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="LipRead Studio API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/jobs", StaticFiles(directory=str(JOBS_DIR), check_dir=False), name="jobs")


@app.get("/health")
def health():
    model_path = ROOT / "models" / "word_classifier.pt"
    return {
        "status": "ok",
        "model_ready": model_path.exists(),
        "model_path": str(model_path),
    }


@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(JOBS_DIR)) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await video.read())

    job = create_job(temp_path)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job.status,
        "progress": job.progress,
        "frames_done": job.frames_done,
        "total_frames": job.total_frames,
        "error": job.error,
    }


@app.get("/frames/{job_id}")
def frames(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"frames": job.frame_urls}


@app.get("/predict/{job_id}")
def predict(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Processing failed")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="Job still processing")
    return {"prediction": job.prediction, "confidence": job.confidence}


@app.post("/predict")
async def predict_direct(video: UploadFile = File(...)):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await video.read())
    try:
        from word_predict import predict_from_video

        prediction = predict_from_video(str(temp_path))
        return {"prediction": prediction}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()
