"""Local capture-to-mesh API."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
import csv
from io import BytesIO
import os
from pathlib import Path
import shutil
import statistics
from time import time
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
import pynvml as nvml

from .capture import ROOT
from .scheduler import Scheduler, new_job_id


MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 32_000_000


async def sample_gpu(app):
    initialized = False
    try:
        nvml.nvmlInit()
        initialized = True
        device = nvml.nvmlDeviceGetHandleByIndex(int(os.getenv("NGP_GPU_INDEX", "0")))
        while True:
            memory = nvml.nvmlDeviceGetMemoryInfo(device)
            power = None
            with suppress(nvml.NVMLError_NotSupported):
                power = nvml.nvmlDeviceGetPowerUsage(device) / 1000
            app.state.gpu.append({
                "time": time(), "job": app.state.scheduler.active,
                "utilization": nvml.nvmlDeviceGetUtilizationRates(device).gpu,
                "memory_mib": round(memory.used / 2**20, 1),
                "total_mib": round(memory.total / 2**20, 1),
                "power_w": power,
            })
            await asyncio.sleep(1)
    except nvml.NVMLError as error:
        app.state.gpu_error = str(error)
    finally:
        if initialized:
            nvml.nvmlShutdown()


@asynccontextmanager
async def lifespan(app):
    app.state.scheduler = Scheduler(Path(os.getenv("NGP_JOBS_DIR", ROOT / "service/jobs")).resolve())
    app.state.gpu = deque(maxlen=180)
    app.state.gpu_error = None
    app.state.scheduler.wakeup.set()
    tasks = [asyncio.create_task(app.state.scheduler.work()), asyncio.create_task(sample_gpu(app))]
    yield
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="NeRF2Mesh Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=os.getenv("NGP_WEB_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"],
)


@app.get("/api/jobs")
async def list_jobs():
    scheduler = app.state.scheduler
    return {"jobs": sorted(scheduler.jobs.values(), key=lambda job: job["submitted_at"], reverse=True),
            "policy": scheduler.policy, "paused": scheduler.paused, "active": scheduler.active}


class QueueSettings(BaseModel):
    policy: Literal["fifo", "sjf"]
    paused: bool = False


@app.put("/api/queue")
async def configure_queue(settings: QueueSettings):
    app.state.scheduler.configure(settings.policy, settings.paused)
    return settings


def normalize_photo(data, output):
    try:
        with Image.open(BytesIO(data)) as photo:
            if photo.format not in {"JPEG", "PNG", "WEBP"} or photo.width * photo.height > MAX_PIXELS:
                raise ValueError("Use JPEG, PNG or WebP files up to 32 megapixels.")
            image = ImageOps.exif_transpose(photo).convert("RGB")
            image.thumbnail((1920, 1920))
            image.save(output, "JPEG", quality=94)
            return image.width * image.height
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("A photo could not be decoded.") from error


@app.post("/api/jobs", status_code=201)
async def upload(
    photos: Annotated[list[UploadFile], File()],
    label: Annotated[str, Form(max_length=80)] = "New capture",
    resolution: Annotated[int, Form()] = 256,
    steps: Annotated[int, Form(ge=100, le=35000)] = 3500,
):
    if not 3 <= len(photos) <= 120:
        raise HTTPException(422, "Choose between 3 and 120 overlapping photos.")
    if resolution not in {128, 256, 512}:
        raise HTTPException(422, "Mesh resolution must be 128, 256, or 512.")
    scheduler = app.state.scheduler
    job_id = new_job_id()
    directory = scheduler.root / job_id
    images = directory / "images"
    images.mkdir(parents=True)
    pixels = 0
    try:
        total = 0
        for index, photo in enumerate(photos):
            data = await photo.read(MAX_BYTES + 1)
            total += len(data)
            if len(data) > MAX_BYTES or total > 500 * 1024 * 1024:
                raise ValueError("Each photo must be under 16 MiB and the full capture under 500 MiB.")
            pixels += await asyncio.to_thread(normalize_photo, data, images / f"photo-{index:04}.jpg")
        return scheduler.add(job_id, label.strip() or "New capture", len(photos), pixels, resolution, steps)
    except ValueError as error:
        shutil.rmtree(directory)
        raise HTTPException(422, str(error)) from error


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = app.state.scheduler.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Capture not found.")
    return job


@app.get("/api/jobs/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str):
    job = await get_job(job_id)
    if name == "worker.log" or (job["status"] == "complete" and name in ["mesh.glb", *job["frames"]]):
        path = app.state.scheduler.root / job_id / name
        if path.is_file():
            media_type = "model/gltf-binary" if name == "mesh.glb" else None
            return FileResponse(path, media_type=media_type, filename=name if name == "mesh.glb" else None)
    raise HTTPException(404, "Artifact not found.")


@app.get("/api/gpu")
async def gpu():
    return {"samples": list(app.state.gpu), "error": app.state.gpu_error}


@app.get("/api/benchmarks")
def benchmarks():
    groups = {}
    recorded = {}
    columns = ("experiment", "scene", "resolution", "variant", "metric", "unit")
    for path in sorted((ROOT / "results").glob("*.csv")):
        recorded[path.name] = path.stat().st_mtime
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if any(column not in row for column in columns):
                    continue
                try:
                    value = float(row["value"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not (-float("inf") < value < float("inf")):
                    continue
                key = (path.name, *(row[column] for column in columns))
                groups.setdefault(key, []).append(value)
    return [
        dict(zip(("source", *columns), key), recorded_at=recorded[key[0]], median=statistics.median(values),
             mean=statistics.fmean(values), minimum=min(values), maximum=max(values), runs=len(values))
        for key, values in groups.items()
    ]
