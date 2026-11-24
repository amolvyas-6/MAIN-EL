"""Submit the same capture batch to FIFO and shortest-job-first queues."""

import argparse
import csv
import json
import mimetypes
from pathlib import Path
import time
from urllib.request import Request, urlopen
import uuid


def request(base, path, method="GET", data=None, content_type="application/json"):
    with urlopen(Request(base + path, method=method, data=data, headers={"Content-Type": content_type}), timeout=180) as response:
        return json.load(response)


def upload(base, directory, steps, resolution):
    boundary = uuid.uuid4().hex
    body = bytearray()

    def field(name, data, filename=None):
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"".encode())
        if filename:
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            body.extend(f"; filename=\"{filename}\"\r\nContent-Type: {mime}".encode())
        body.extend(b"\r\n\r\n" + data + b"\r\n")

    field("label", directory.name.encode())
    field("steps", str(steps).encode())
    field("resolution", str(resolution).encode())
    photos = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for index, path in enumerate(photos):
        field("photos", path.read_bytes(), f"photo-{index}{path.suffix.lower()}")
    body.extend(f"--{boundary}--\r\n".encode())
    return request(base, "/api/jobs", "POST", bytes(body), f"multipart/form-data; boundary={boundary}")


def configure(base, policy, paused):
    return request(base, "/api/queue", "PUT", json.dumps({"policy": policy, "paused": paused}).encode())


def experiment(base, policy, directories, steps, resolution, timeout):
    state = request(base, "/api/jobs")
    if any(job["status"] not in {"complete", "failed"} for job in state["jobs"]):
        raise RuntimeError("Start the scheduling comparison with an empty pending queue.")
    configure(base, policy, True)
    try:
        captures = [upload(base, directory, steps, resolution) for directory in directories]
    finally:
        configure(base, policy, False)
    start = time.time()
    samples = {}
    rows = []
    while True:
        jobs = [request(base, f"/api/jobs/{capture['id']}") for capture in captures]
        for sample in request(base, "/api/gpu")["samples"]:
            if sample["time"] >= start:
                samples[sample["time"]] = sample
        if all(job["status"] in {"complete", "failed"} for job in jobs):
            break
        if time.time() - start > timeout:
            raise TimeoutError(f"{policy} experiment did not finish within {timeout}s.")
        time.sleep(2)

    dispatch = sorted(jobs, key=lambda job: job["started_at"])
    finished = {job["id"]: job for job in jobs if job["status"] == "complete"}
    for index, job in enumerate(dispatch):
        key = ["scheduling", job["label"], f"{resolution}³", policy, index]
        print(f"{policy} {index + 1}: {job['label']} ({job['status']})", flush=True)
        if job["id"] not in finished:
            rows.append([*key, "failed", 1, "boolean"])
            continue
        rows.extend([
            [*key, "dispatch_order", index + 1, "position"],
            [*key, "queue_wait", max(0, job["started_at"] - start), "s"],
            [*key, "turnaround", job["finished_at"] - start, "s"],
            [*key, "execution", job["finished_at"] - job["started_at"], "s"],
        ])
    for index, sample in enumerate(samples.values()):
        key = ["scheduling", "batch", f"{resolution}³", policy, index]
        rows.extend([[*key, "gpu_utilization", sample["utilization"], "%"],
                     [*key, "gpu_memory", sample["memory_mib"], "MiB"]])
        if sample["power_w"] is not None:
            rows.append([*key, "gpu_power", sample["power_w"], "W"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path, help="Photo directories, submitted in order.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("results/scheduling_capture.csv"))
    args = parser.parse_args()
    if len(args.captures) < 2 or len({path.name for path in args.captures}) != len(args.captures):
        parser.error("Provide at least two capture directories with distinct names.")
    policy = request(args.url, "/api/jobs")["policy"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["experiment", "scene", "resolution", "variant", "run", "metric", "value", "unit"])
            for name in ("fifo", "sjf"):
                writer.writerows(experiment(args.url, name, args.captures, args.steps, args.resolution, args.timeout))
                handle.flush()
    finally:
        configure(args.url, policy, False)
    print(f"Measurements saved to {args.output}")


if __name__ == "__main__":
    main()
