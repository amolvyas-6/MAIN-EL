# NeRF2Mesh-GPU

NeRF2Mesh-GPU is a course project built on [NVIDIA Instant-NGP](https://github.com/NVlabs/instant-ngp). It takes a set of overlapping photos, reconstructs a neural radiance field (NeRF), extracts a polygon mesh, and provides a browser interface for viewing and measuring the results. The project adds GPU distance transforms for safer ray skipping, occupancy-guided sparse marching cubes, and a local capture-to-mesh service.

The project is intended for an NVIDIA CUDA GPU. Linux is the verified development platform; Windows is a supported target for the CUDA application, but this fork's full build and service have not been validated on Windows. macOS cannot run the CUDA reconstruction backend because current Macs do not provide an NVIDIA CUDA GPU. The browser interface can run on macOS when it connects to a backend on a supported Linux or Windows machine.

## What it does

1. **Reconstructs a scene:** upload 3–120 overlapping JPEG, PNG, or WebP photos. COLMAP estimates camera poses, then headless Instant-NGP trains a NeRF.
2. **Extracts a mesh:** occupancy-guided sparse marching cubes produces a GLB file. The web viewer displays it shaded or as wireframe and lets you download it.
3. **Renders the NeRF:** the job page includes selected views rendered from the trained model.
4. **Measures and compares:** the Measurements page presents recorded CUDA timings, mesh quality and memory, distance-transform accuracy, and FIFO/SJF scheduling results. The app reads CSV files from `results/`; it does not invent missing results.
5. **Explores GPU scheduling:** the service queues captures under FIFO or shortest-job-first (SJF) and displays NVIDIA GPU utilization, memory, and power when NVML provides them.

For the implementation rationale and course mapping, see [Implementation and course rationale](IMPLEMENTATION_README.md). For recorded results and their limits, see [docs/RESULTS.md](docs/RESULTS.md). A classroom walkthrough is in [docs/DEMO.md](docs/DEMO.md), and the CUDA source map and thread mappings are in [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

### Linux (verified)

- 64-bit Linux, NVIDIA GPU and a working NVIDIA driver (`nvidia-smi` must work).
- CUDA Toolkit, CMake 3.21+, Git, Node.js/npm, Python 3.12, and a compatible C++ compiler.
- For the recorded Arch Linux environment, use GCC/G++ 14, CUDA 13.x, and `TCNN_CUDA_ARCHITECTURES=86` for an RTX 3070 Ti Laptop GPU. Other NVIDIA cards need their matching CUDA architecture. Use `-j 8` or less for compilation on a laptop.
- COLMAP reconstruction uses PyCOLMAP, installed from `service/requirements.txt`. Python bindings and the NVIDIA management library (NVML) are also needed by the service.

### Windows (instructions; this checkout's full path is unverified)

- 64-bit Windows, NVIDIA GPU and driver, CUDA Toolkit compatible with the selected Visual Studio release, Visual Studio 2022 with the C++ workload, CMake, Git, Python 3.12, and Node.js/npm.
- The upstream Instant-NGP project supports Windows CUDA builds. This fork's Linux-specific compiler and venv commands below do not apply verbatim on Windows. Use a Visual Studio Developer PowerShell, configure CMake with the Visual Studio generator and your Python executable, and build with CMake. PyCOLMAP/NVML wheel and service behavior can vary by Windows/Python/CUDA combination, so verify those dependencies before relying on capture jobs.
- The Windows commands below show the intended setup flow; they are not a claim that this fork has been built or end-to-end tested on Windows.

### macOS

The CUDA/C++ reconstruction, ray-skipping, sparse-mesh, COLMAP-to-NeRF worker and live NVML telemetry require a supported NVIDIA CUDA host. They do not run natively on current macOS hardware. You can run the Next.js viewer locally and point it at an API running on a supported GPU host by setting `NEXT_PUBLIC_NGP_API` before building the frontend and allowing that origin in the backend's `NGP_WEB_ORIGINS` setting. Browser access to the remote API must be permitted by your network and CORS configuration.

## Get the source

Clone the repository with its CUDA and other submodules:

```bash
git clone --recursive https://github.com/amolvyas-6/MAIN-EL.git
cd MAIN-EL
```

If you already cloned it without submodules, run `git submodule update --init --recursive` from the repository directory.

## Build and run on Linux

Commands below are run from the repository root. The compiler selection is the one used by the verified Arch Linux setup; adapt the compiler executable names on another Linux distribution, while keeping a CUDA-compatible host compiler.

### 1. Build the CUDA application and Python bindings

```bash
cmake -S . -B build \
  -DNGP_BUILD_WITH_GUI=OFF \
  -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DTCNN_CUDA_ARCHITECTURES=86 \
  -DPython_EXECUTABLE="$(command -v python3.12)"
cmake --build build --config RelWithDebInfo -j 4
```

Change `TCNN_CUDA_ARCHITECTURES=86` to match your GPU if it is not an RTX 3070 Ti / compute capability 8.6. CUDA compilation can use substantial host RAM. If changing compiler after configuring, remove the CMake build directory and configure again.

### 2. Install Python and web dependencies

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r service/requirements.txt
npm ci --prefix service/web
npm run build --prefix service/web
```

### 3. Start the API and web UI

Open two terminals in the repository root. Keep the API terminal running:

```bash
.venv/bin/python -m uvicorn service.main:app --host 127.0.0.1 --port 8000
```

In the second terminal:

```bash
npm run start --prefix service/web -- --hostname 127.0.0.1
```

Open <http://127.0.0.1:3000>. To develop the frontend with live reload, use `npm run dev --prefix service/web -- --hostname 127.0.0.1` instead of the production start command.

The worker expects the Python bindings in `build/`, keeps uploads and job artifacts under `service/jobs/`, reads benchmark CSV files from `results/`, and selects GPU 0 by default. Optional environment settings are `NGP_BUILD_DIR`, `NGP_JOBS_DIR`, and `NGP_GPU_INDEX`.

### 4. Create a capture

In the Studio, select 3–120 overlapping photos (JPEG, PNG, or WebP), enter a label, select a mesh grid of 128³, 256³, or 512³, and choose training steps from 100 to 35,000. Press **Reconstruct**. COLMAP needs enough overlap to register at least three photos. The default worker runs COLMAP feature extraction and exhaustive matching on the CPU; larger photo sets can take several minutes. Completed jobs have shaded, wireframe and NeRF-view modes plus a downloadable `mesh.glb`.

For a quick reproducible input set, use the bundled `data/nerf/fox/images/` photographs. For input capture tips and the headless CLI, see [docs/DEMO.md](docs/DEMO.md).

### Headless command-line use

With the build and Python environment above, train the bundled scene and save a snapshot:

```bash
PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python scripts/run.py --scene data/nerf/fox --n_steps 5000 \
  --save_snapshot results/fox.ingp
```

Export a GLB using exact distance-guided rendering and sparse mesh extraction:

```bash
PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python scripts/run.py --scene data/nerf/fox \
  --load_snapshot results/fox.ingp --n_steps 0 \
  --use-distance-skip --sparse-mesh --mesh-block-size 8 \
  --save_mesh results/fox.glb --marching_cubes_res 128
```

The acceleration flags are opt-in. Without `--use-distance-skip` rendering follows the baseline stepping path; without `--sparse-mesh`, extraction uses dense marching cubes. `--mesh-block-size` accepts 8 or 16. Large dense grids can exhaust GPU memory; use the measured outcomes in `docs/RESULTS.md` as context, not as a guarantee for another GPU.

## Windows setup and launch

Open **Developer PowerShell for Visual Studio 2022** after installing the requirements above. Clone recursively as shown earlier, then from the repository root configure and build with the Visual Studio generator. Replace the Python path with the location of your Python 3.12 installation and adjust CUDA architecture for your GPU:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DNGP_BUILD_WITH_GUI=OFF `
  -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DPython_EXECUTABLE="C:/Path/To/Python312/python.exe"
cmake --build build --config RelWithDebInfo --parallel 4
```

Create a virtual environment and install dependencies (replace the Python launcher if `py -3.12` is not available):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r service/requirements.txt
npm ci --prefix service/web
npm run build --prefix service/web
```

Start the API and frontend from separate PowerShell terminals in the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn service.main:app --host 127.0.0.1 --port 8000
```

```powershell
npm run start --prefix service/web -- --hostname 127.0.0.1
```

Then open <http://127.0.0.1:3000>. These are the expected CMake/Windows commands, but the full CUDA build and photo service on Windows have not been verified for this fork. If configuration or dependency installation fails, use the linked Linux setup as the validated environment.

## Demonstration and measurements

- **Ray skipping:** Measurements → Ray skipping compares the same scene and camera with no skipping and exact distance-guided skipping. Show both frame time and empty-space advances; fewer advances do not always mean a faster frame on every GPU.
- **Sparse mesh:** Measurements → Mesh extraction compares dense and sparse extraction at the same resolution. On the recorded 8 GB laptop GPU, dense extraction ran out of memory at 768³ while sparse completed. At 512³, both output meshes had matching triangle counts and zero sampled Chamfer distance.
- **Distance-transform algorithms:** the exact separable transform is the safe field used for skipping. Jump flooding is included as an approximate speed/accuracy comparison and is not used to drive ray jumps.
- **Scheduling:** switch between FIFO and SJF in the queue and compare the recorded multi-job experiment. The worker runs one reconstruction at a time on the GPU.
- **CGVR:** inspect the shaded mesh, wireframe, rendered NeRF training views, and GLB in the browser.

Numbers are machine- and workload-dependent. Consult [docs/RESULTS.md](docs/RESULTS.md) and the raw CSVs under `results/` for the measured conditions. Do not present a planned target as a measured result.

## Project contents

| Path | Contents |
| --- | --- |
| `src/distance_transform.cu` and `include/neural-graphics-primitives/distance_transform.h` | Exact separable EDT and jump-flood distance fields |
| `src/sparse_marching_cubes.cu` and matching header | Occupancy culling, scans, sparse extraction and GLB mesh data |
| `src/testbed_nerf.cu` | Integration with NeRF density-grid update, rendering and mesh paths |
| `service/` | FastAPI API, reconstruction worker, FIFO/SJF scheduling and GPU telemetry |
| `service/web/` | Next.js application, measurements interface and Three.js viewer |
| `bench/`, `tests/`, `results/` | Benchmark scripts, correctness fixtures and recorded CSV measurements |
| `docs/` | Demo steps, source/design notes, environment and results |

## Credits and upstream

This repository is based on NVIDIA's [Instant-NGP](https://github.com/NVlabs/instant-ngp), the 2022 neural graphics primitives project by Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. See the upstream project and [license](LICENSE) for original work and license details. New project-specific implementation and experiments are described in [IMPLEMENTATION_README.md](IMPLEMENTATION_README.md).
