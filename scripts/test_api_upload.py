import json
import time
import urllib.request
import uuid
from pathlib import Path


def post_multipart(url, field, filename, data, content_type="video/mp4"):
    boundary = uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        data,
        f"--{boundary}--".encode(),
        b"",
    ]
    payload = b"\r\n".join(body)
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


video = Path("evaluation/samples/sentence_demo.mp4").read_bytes()
upload = post_multipart("http://127.0.0.1:8000/upload", "video", "sentence_demo.mp4", video)
job_id = upload["job_id"]
for _ in range(60):
    with urllib.request.urlopen(f"http://127.0.0.1:8000/status/{job_id}") as resp:
        status = json.loads(resp.read())
    if status["status"] in ("done", "failed"):
        break
    time.sleep(0.5)

with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
    print("health", resp.read().decode())
with urllib.request.urlopen(f"http://127.0.0.1:8000/predict/{job_id}") as resp:
    print("predict", resp.read().decode())
