# NeRF2Mesh Studio

NeRF2Mesh Studio reconstructs a NeRF and an interactive 3D mesh from overlapping
photos. The local FastAPI worker runs COLMAP camera recovery, Instant-NGP
training, sparse marching cubes, and NeRF rendering. Next.js shows the capture
queue and recorded measurements. Three.js loads the GLB and supports orbit,
zoom, and wireframe inspection.

## What we improved

CUDA launches many GPU workers, called threads. Each thread gets its own piece
of the scene or image to work on.

In the snippets, `__global__` marks a function launched on the GPU. `void`
means it returns without a value. These GPU functions write their results
into arrays such as `output` or `flags`. `int` is a whole number, `uint32_t`
is a 32-bit non-negative whole number, and `float` can hold a fraction.
`*` means the variable points to data in memory, and `const` marks data the
function reads without changing it.

### 1. Euclidean distance transform (EDT)

Imagine the scene as a grid of tiny cubes. Before rendering, we make a table
that tells each cube how far its center is from the center of the nearest
occupied cube. Many rays can reuse that table. [src/distance_transform.cu](../src/distance_transform.cu)

```cuda
__global__ void transform_axis(int lines, int side, int axis, const int* input, int* output)
```

`distance_transform` launches `transform_axis` three times, once for each
direction (X, Y, and Z). Each pass adds the squared distance along that
direction and keeps the smallest total for each cube. If an occupied cube is
three steps away in X and four in Y at the same height, its squared distance
is `3² + 4² = 25`. Taking the square root gives a straight-line distance of
`5` steps. This approach lets GPU threads work on different lines of cubes
at the same time during each pass.

- `lines` is how many lines of cubes this pass processes.
- `side` is how many cubes fit along one side of the grid.
- `axis` selects the direction. `0` is X, `1` is Y, and `2` is Z.
- `input` holds the starting grid values on the first pass and the previous
  pass's squared distances after that. `output` receives this pass's squared
  distances. The square root is taken after all three passes.

### 2. Distance-guided ray skipping

To draw a pixel, we trace a ray from the camera into the scene and check
points along its path. The GPU traces many rays in parallel. [Skip logic](../include/neural-graphics-primitives/nerf_device.cuh) · [Fused renderer](../include/neural-graphics-primitives/fused_kernels/render_nerf.cuh)

```cuda
float safe = distance_skip_bound(
    pos, ray.d, idir, mip, lookup_mip, marching.distances, next_boundary
);
```

In an empty cube, `distance_skip_bound` reads the distance table and leaves
a safety margin because the distances are measured from cube centers. If the
next cube face is one unit away and the safe limit is five units, the renderer
can jump directly to a sample four units ahead, if there is one there. The
checks in between are skipped. The usual move to the next cube is still used
when the safe limit is too short.

- `safe` holds the upper limit returned by `distance_skip_bound` for a jump
  through empty space.
- `pos` is the ray's current 3D position. `ray.d` is its direction.
- `idir` is `1` divided by each part of `ray.d`. It helps calculate where the
  ray meets a grid boundary.
- `mip` selects the grid level used to read the distance table. `lookup_mip`
  is the original grid level used to check occupancy. Its boundaries help
  limit the jump. Each level covers a different amount of space.
- `marching.distances` is the distance table built by the EDT.
- `next_boundary` is the distance to the next cube face along the ray. A
  longer jump must beat this distance.

### 3. Sparse marching cubes

A mesh is a surface made of triangles. We divide the scene into chunks so the
GPU can decide where to look for that surface. [src/sparse_marching_cubes.cu](../src/sparse_marching_cubes.cu)

```cuda
__global__ void occupancy_flags(
    uint32_t count, Lattice lattice, const float* occupancy, uint32_t max_mip, uint32_t* flags
)
```

`occupancy_flags` uses the model's existing map of occupied space. One GPU
thread checks each chunk. If three chunks get flags `[0, 1, 0]`, only the middle
chunk goes on to the more expensive density calculation. This saves the work
of calculating densities in empty chunks and the memory to store them. Later
stages count triangles where density crosses a cutoff and use a running total
to give each cell its own space to write them.

- `count` is the total number of chunks to check.
- `lattice` is a `Lattice` object that describes the grid, chunk size, and
  where the grid sits in the scene.
- `occupancy` holds the stored density values used to check for empty space.
- `max_mip` is the largest grid level the function can check.
- `flags` stores one answer per chunk. `1` means keep it for a closer check.
  `0` means skip it.

### How we schedule capture jobs

The service finishes one capture job before picking the next. **First In,
First Out (FIFO)** picks the job that arrived first. **Shortest Job First
(SJF)** picks the waiting job with the smallest estimated workload. The estimate
uses the number and size of the photos, the training steps, and the requested
mesh resolution. Ties go to the earlier job. [service/scheduler.py](../service/scheduler.py)

```python
job = min(waiting, key=POLICIES[self.policy])
```

If a large capture arrives first and a small capture arrives second, FIFO
chooses the large one when both are waiting. SJF lets the small one go first
if it has the lower estimated workload.

- `waiting` is the list of jobs still in the queue. `job` is the one chosen
  to run next.
- `POLICIES` holds the ranking rules. `self.policy` is the current setting
  (`fifo` or `sjf`). `POLICIES[self.policy]` looks up the matching rule.
- `min` picks the job that ranks first. `key=` tells it which rule to use
  when ranking the jobs.

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
   mesh, reveal the triangles in Wireframe, and scrub through NeRF views rendered
   from camera poses between nearby capture photos.
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
