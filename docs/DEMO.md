# NeRF2Mesh Studio

NeRF2Mesh Studio reconstructs a NeRF and an interactive 3D mesh from overlapping
photos. The local FastAPI worker runs COLMAP camera recovery, Instant-NGP
training, sparse marching cubes, and NeRF rendering. Next.js shows the capture
queue and recorded measurements. Three.js loads the GLB and supports orbit,
zoom, and wireframe inspection.

## Build and start

Run from the repository root on the NVIDIA machine. The build uses the existing
Python bindings and the testbed GPU path.

```bash
git submodule update --init --recursive
cmake -S . -B build -DNGP_BUILD_WITH_GUI=OFF \
  -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DTCNN_CUDA_ARCHITECTURES=86 \
  -DPython_EXECUTABLE="$(command -v python3.12)"
cmake --build build --config RelWithDebInfo -j 4

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r service/requirements.txt
npm ci --prefix service/web
npm run build --prefix service/web
```

Start the API from the repo root:

```bash
.venv/bin/python -m uvicorn service.main:app --host 127.0.0.1 --port 8000
```

Start the viewer in a second terminal, also from the repo root:

```bash
npm run start --prefix service/web -- --hostname 127.0.0.1
```

Open [NeRF2Mesh Studio](http://127.0.0.1:3000). The worker runs camera recovery
through PyCOLMAP, so the capture service uses a Python wheel for COLMAP. Job
inputs, logs, snapshots, and exports remain under `service/jobs/`.
`NGP_BUILD_DIR`, `NGP_JOBS_DIR`, and `NGP_GPU_INDEX` configure the worker.
The service is designed for a single API process with a single GPU worker.

## Classroom walkthrough

1. Upload overlapping photos with a varied set of camera angles. The bundled
   `data/nerf/fox/images/` directory is a reproducible photo source. Use the
   `128³` mesh grid and a short training run for an in-class capture. Show a
   completed capture while the new job is queued or training.
2. Open the completed job. Use the tabs above the preview to orbit the shaded
   mesh, reveal the triangles in Wireframe, and scrub through NeRF views.
   Use **Download GLB** to save the mesh.
3. Open Measurements. **Ray skipping** shows **No skipping** and
   **Distance-guided skipping** side by side. Describe the change in
   empty-space advances separately from the change in frame time.
   Use **Mesh extraction** at 512³ for the dense and sparse speed and memory
   comparison. Open **Measurement details** for the ranges and source CSV.
4. Show the same capture batch under FIFO and shortest-job-first. Point out
   the dispatch order, queue wait, turnaround, and GPU utilization trace.
5. Relate the EDT lines, voxel-wide jump propagation, scans, and reductions
   to PAGP. Relate sample placement, density-to-alpha compositing, isosurface
   interpolation, vertex normals, and orbit viewing to CGVR.

## Run the comparisons

Build and set the Python import path first. Keep other GPU jobs idle during the
CUDA-event comparisons. Every timed series discards warm-up runs.

```bash
export PYTHONPATH="$PWD/build"
mkdir -p results
.venv/bin/python tests/distance_transform.py

.venv/bin/python bench/synthetic_scene.py results/spheres
.venv/bin/python scripts/run.py --scene data/nerf/fox --n_steps 5000 \
  --save_snapshot results/fox.ingp
.venv/bin/python scripts/run.py --scene results/spheres/train.json --n_steps 5000 \
  --save_snapshot results/spheres.ingp

.venv/bin/python bench/compare.py render --scene data/nerf/fox \
  --snapshot results/fox.ingp --name fox
.venv/bin/python bench/compare.py render --scene results/spheres/train.json \
  --snapshot results/spheres.ingp --test-scene results/spheres/test.json --name spheres
.venv/bin/python bench/compare.py render --scene data/nerf/fox \
  --snapshot results/fox.ingp --name fox --no-jit

.venv/bin/python bench/compare.py mesh --scene data/nerf/fox \
  --snapshot results/fox.ingp --name fox

.venv/bin/python scripts/run.py --scene data/nerf/fox \
  --load_snapshot results/fox.ingp --n_steps 0 --sparse-mesh \
  --use-distance-skip --save_mesh results/fox.glb --marching_cubes_res 128
```

Omitting `--use-distance-skip` and `--sparse-mesh` keeps the original render
and extraction paths. `--mesh-block-size 16` selects larger sparse tiles.
The render benchmark times two modes. No skipping checks every sample
position for occupancy, and distance-guided skipping jumps across empty
space. Fixed-grid voxel stepping serves as an image regression check.
Each mode uses the same saved scene, view and resolution. Timed runs
alternate the mode order and record the laptop power source when
available. The run is saved as a `*_no_skip.csv` file. Use a distinct
`--name` to keep a separate run when changing power conditions.

With the API running and its pending queue empty, run the scheduler comparison
with capture directories in a fixed submission order:

```bash
.venv/bin/python bench/scheduling.py path/to/large path/to/medium \
  path/to/small path/to/second-medium
```

To reproduce the recorded Fox scheduling workload, make the four overlapping
subsets from the bundled source photos first:

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
from PIL import Image, ImageOps

source = sorted(Path("data/nerf/fox/images").glob("*.jpg"))
for count in (24, 16, 12, 20):
    directory = Path("results/scheduling-small") / f"capture-{count}"
    directory.mkdir(parents=True, exist_ok=True)
    for path in source[:count]:
        with Image.open(path) as photo:
            image = ImageOps.exif_transpose(photo)
            image.thumbnail((800, 800))
            image.save(directory / path.name, quality=94)
PYTHON
.venv/bin/python bench/scheduling.py results/scheduling-small/capture-24 \
  results/scheduling-small/capture-16 results/scheduling-small/capture-12 \
  results/scheduling-small/capture-20 --steps 250
```

The comparison loads the same capture batch into a paused queue for each policy,
then releases the batch. Turnaround and queue wait start at release. The CSV
contains observed dispatch orders, job durations, failed jobs, and NVML
samples. Successful jobs appear in the viewer for inspection.

The measurement files use
`experiment,scene,resolution,variant,run,metric,value,unit`. See
[Results](RESULTS.md) for the measured outcomes and
[Design](DESIGN.md) for the CUDA thread mappings.
