"""Persistent single-GPU queue. Policy changes affect waiting jobs only."""

import asyncio
import json
from pathlib import Path
import time
import uuid

from . import capture


POLICIES = {
    "fifo": lambda job: (job["submitted_at"], job["id"]),
    "sjf": lambda job: (job["estimated_work"], job["submitted_at"], job["id"]),
}


class Scheduler:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        for path in self.root.glob("*/job.json"):
            job = json.loads(path.read_text())
            self.jobs[job["id"]] = job
            if job["status"] in {"poses", "training", "exporting"}:
                self.update(job, status="failed", error="Worker restarted during this job.", finished_at=time.time())
        self.policy = "fifo"
        self.paused = False
        self.wakeup = asyncio.Event()
        self.active = None

    def update(self, job, **changes):
        job.update(changes)
        path = self.root / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(path)

    def add(self, job_id, label, images, pixels, resolution, steps):
        job = {
            "id": job_id, "label": label, "images": images, "pixels": pixels,
            "resolution": resolution, "steps": steps, "status": "queued", "submitted_at": time.time(),
            # Image count drives matching pairs. Total pixels estimates feature extraction.
            "estimated_work": pixels / 1e6 + images * (images - 1) / 2 + steps / 100 + (resolution / 128) ** 3,
        }
        self.update(job)
        self.jobs[job_id] = job
        self.wakeup.set()
        return job

    def configure(self, policy, paused):
        self.policy = policy
        self.paused = paused
        self.wakeup.set()

    async def work(self):
        while True:
            await self.wakeup.wait()
            self.wakeup.clear()
            while not self.paused:
                waiting = [job for job in self.jobs.values() if job["status"] == "queued"]
                if not waiting:
                    break
                job = min(waiting, key=POLICIES[self.policy])
                self.active = job["id"]
                directory = self.root / job["id"]
                try:
                    self.update(job, status="poses", started_at=time.time(), policy=self.policy)
                    registered = await asyncio.to_thread(capture.build_poses, directory)
                    self.update(job, status="training", registered=registered)
                    await asyncio.to_thread(capture.train, directory, job["steps"])
                    self.update(job, status="exporting")
                    artifacts = await asyncio.to_thread(capture.export, directory, job["resolution"])
                    self.update(job, status="complete", finished_at=time.time(), **artifacts)
                except Exception as error:
                    self.update(job, status="failed", finished_at=time.time(), error=str(error))
                finally:
                    self.active = None


def new_job_id():
    return uuid.uuid4().hex
