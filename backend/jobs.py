import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.preprocess import mouth_frames_to_jpegs, process_video

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "data" / "jobs"


@dataclass
class JobState:
    job_id: str
    video_path: Path
    status: str = "queued"
    progress: float = 0.0
    frames_done: int = 0
    total_frames: int = 0
    frame_urls: list[str] = field(default_factory=list)
    prediction: str = ""
    confidence: float = 0.0
    method: str = ""
    error: str = ""


_jobs: dict[str, JobState] = {}
_lock = threading.Lock()


def create_job(video_path: Path) -> JobState:
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    stored_video = job_dir / f"upload{video_path.suffix or '.mp4'}"
    shutil.copy2(video_path, stored_video)
    job = JobState(job_id=job_id, video_path=stored_video)
    with _lock:
        _jobs[job_id] = job
    thread = threading.Thread(target=_process_job, args=(job_id,), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> JobState | None:
    with _lock:
        return _jobs.get(job_id)


def _process_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return

    job_dir = JOBS_DIR / job_id
    frames_dir = job_dir / "frames"
    try:
        job.status = "processing"
        job.progress = 0.1

        mouth_arr, npy_path = process_video(
            str(job.video_path),
            save_dir=str(job_dir),
            max_frames=75,
            output_size=(96, 96),
        )
        job.total_frames = len(mouth_arr)
        job.frames_done = len(mouth_arr)
        job.progress = 0.7

        jpeg_paths = mouth_frames_to_jpegs(mouth_arr, str(frames_dir))
        job.frame_urls = [
            f"/jobs/{job_id}/frames/{Path(path).name}" for path in jpeg_paths
        ]
        job.progress = 0.85

        prediction, confidence, method = _predict_sentence(mouth_arr, job.video_path)
        job.prediction = prediction
        job.confidence = confidence
        job.method = method
        job.status = "done"
        job.progress = 1.0
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress = 1.0
    finally:
        if job.video_path.exists():
            try:
                job.video_path.unlink()
            except OSError:
                pass


def _predict_sentence(mouth_arr, video_path):
    from sentence_predict import predict_sentence_ctc

    sentence, confidence = predict_sentence_ctc(mouth_arr)
    return sentence, confidence, "ctc"
