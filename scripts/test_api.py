import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "evaluation" / "samples" / "id2_vcd_swwp2s.mpg"
API = "http://127.0.0.1:8000"


def post_file(url, path):
    boundary = "----bound"
    data = Path(path).read_bytes()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="video"; filename="{Path(path).name}"\r\n'
        f"Content-Type: video/mpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.loads(urllib.request.urlopen(req).read())


def main():
    health = json.loads(urllib.request.urlopen(f"{API}/health").read())
    print("health:", health)
    upload = post_file(f"{API}/upload", SAMPLE)
    print("upload:", upload)
    job_id = upload["job_id"]
    status = {}
    for _ in range(90):
        status = json.loads(urllib.request.urlopen(f"{API}/status/{job_id}").read())
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    print("status:", status)
    predict = json.loads(urllib.request.urlopen(f"{API}/predict/{job_id}").read())
    print("predict:", predict)
    frames = json.loads(urllib.request.urlopen(f"{API}/frames/{job_id}").read())
    print("frames:", len(frames.get("frames", [])))


if __name__ == "__main__":
    main()
