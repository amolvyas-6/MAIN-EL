# AGENTS.md: NeRF2Mesh-GPU

Read this file fully before doing anything. It is the single source of truth for what to build, how to build it and how to report it. If something here conflicts with a guess you want to make, follow this file or ask.

## 1. Project in one paragraph

NeRF2Mesh-GPU extends the Instant-NGP codebase (NVlabs/instant-ngp, CUDA/C++) with two GPU-parallel optimisations and a web service around them:

1. **GPU Euclidean distance transform for ray skipping.** Compute, for every voxel of the occupancy grid, the distance to the nearest occupied voxel. Use it in the ray marcher so a ray can jump across empty space in one step instead of walking cell by cell. The rendered image must stay unchanged.
2. **Occupancy-guided sparse marching cubes.** Cull empty blocks first, evaluate density only where a surface can exist, build triangles with a prefix-sum (stream compaction) pass, export as GLB. It must use less time and memory than dense extraction.
3. **Capture-to-mesh web service.** FastAPI back end runs COLMAP and headless Instant-NGP on uploaded photos, compares GPU job-scheduling policies (FIFO vs shortest-job-first) with live GPU telemetry, and serves results to a three.js viewer.

Motivation: NGP-RT (ECCV 2024) occupancy-distance idea; Felzenszwalb & Huttenlocher (exact separable EDT); Rong & Tan (jump flooding); Dyken et al. (HistoPyramids, scan-based MC compaction).

Courses: Parallel Architecture and GPU Programming (CS372IA) + Computer Graphics and Virtual Reality, RV College of Engineering. Team of 3.

## 2. Ground rules (non-negotiable)

- **No fabricated results.** Every number in a report must come from a command you actually ran and a file in `results/`. If you did not measure it, write "not measured". The Phase-1 deck states: "All outcomes are planned targets. No results have been measured yet."
- **Correctness before speed.** Each optimisation needs a passing correctness test before any benchmark is recorded.
- **Work phase by phase.** Finish a phase, run its acceptance checks, summarise, then stop and wait for approval. Do not start the next phase on your own.
- **Small commits.** One logical change per commit, committed on the current branch (separate phase branches are not required), message format `phaseN: short description`. Never force-push. Never commit datasets, checkpoints or build folders.
- **Do not break the baseline.** New behaviour sits behind flags (for example `--use-distance-skip`, `--sparse-mesh`) so baseline and new paths can be compared in the same binary.
- **The human compiles, runs and measures. The agent does not.** Do NOT run `cmake`, `make`, the build, `instant-ngp`, training, benchmarks or Nsight yourself. You write and edit code, scripts and tests, then give me the exact commands to run (one copy-paste block per step, with the working directory stated). I run them and paste the output back to you. Reading files, grepping and editing are fine. When I paste a compiler error, find the first real error (not the warnings or the last line), explain the cause in plain words, and propose the smallest fix.
- **Ask before** installing system packages, changing the NVIDIA driver/CUDA toolkit, or deleting anything outside the repo.
- Keep explanations of design choices short and in plain words in commit messages and the progress log (`docs/PROGRESS.md`): what changed and why.

## 3. Environment

- OS: Arch Linux laptop. GPU: NVIDIA RTX 3070 Ti Laptop (Ampere, compute capability 8.6, 8 GB VRAM). Use `TCNN_CUDA_ARCHITECTURES=86`.
- 8 GB VRAM matters: a dense 1024³ float density grid is about 4.3 GB, so the dense baseline may fail or be tight at 1024³. Benchmark dense vs sparse at 256, 512, 768 and 1024 and record out-of-memory as a result, not an error to hide.
- Tools: CUDA toolkit, CMake, a CUDA-compatible GCC (Arch's default GCC is often too new for nvcc; use the `gcc12`/`gcc13` package and set `CC`, `CXX`, `CUDAHOSTCXX`), COLMAP, Python 3 venv, Nsight Systems and Nsight Compute, FastAPI, three.js, NVML (`pynvml`).
- Verify actual versions with `nvidia-smi`, `nvcc --version`, `gcc --version` at the start and record them in `docs/ENVIRONMENT.md`.

### Confirmed working build (done by the human)
- Project path: `~/Projects/MAIN-EL` (the fork of instant-ngp, cloned with `--recursive`).
- System: CUDA 13.4 (nvcc 13.4.92, installed at `/opt/cuda`), default GCC is 16.2.1, Python 3.12, target architecture 86 (auto-detected), `TCNN_HALF_PRECISION=ON`.
- **GCC 16 does NOT work**: tiny-cuda-nn's bundled CUTLASS has a typo (`set_slice3x3` instead of `set_slice_3x3`) that GCC 16 rejects as a hard error. **GCC 14 works.**
- Working configure and build commands (run by the human from the repo root):
  ```bash
  cmake -B build -DNGP_BUILD_WITH_GUI=off -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-14
  cmake --build build --config RelWithDebInfo -j 8
  ```
  Use `-j 8` or lower; nvcc jobs use a lot of RAM on a laptop. After changing compiler settings, delete `build/` first because CMake caches the compiler.
- Whenever you give build commands, keep using GCC 14 as above. When you add new `.cu` files, register them in `CMakeLists.txt` and tell me to re-run the configure step, then the build.
- Deprecation warnings such as `long4 is deprecated` come from CUTLASS and are harmless; ignore them.
- `./instant-ngp data/nerf/fox` ran successfully, so the toolchain, GPU and sample data work. (If you need to know whether the GUI is enabled in the current build, ask me; do not assume.)

## 4. Repository layout (target)

```
MAIN-EL/
  src/distance_transform.cu            new: exact EDT + jump flooding kernels
  include/neural-graphics-primitives/distance_transform.h
  src/sparse_marching_cubes.cu         new: block cull, classify, scan, emit
  include/neural-graphics-primitives/sparse_marching_cubes.h
  src/testbed_nerf.cu                  modified: distance-guided stepping (behind flag)
  CMakeLists.txt                       modified: register new .cu files
  tests/                               CPU references + correctness tests
  bench/                               benchmark drivers (CSV output)
  results/                             raw CSVs, Nsight reports, rendered images (gitignored if large)
  service/                             FastAPI app, scheduler, telemetry, viewer (three.js)
  docs/                                PROGRESS.md, ENVIRONMENT.md, DESIGN.md
  AGENTS.md                            this file
```

Instant-NGP details (from memory of the codebase; **verify by reading the code before relying on them**): the NeRF occupancy grid is 128³ per cascade with several cascades (mip levels) stored as a bitfield; ray stepping and "advance to next occupied voxel" logic live in `src/testbed_nerf.cu`; existing marching cubes lives in `src/marching_cubes.cu` and is driven from `src/testbed.cu`; the Python bindings and `scripts/run.py` give headless training, rendering, mesh saving and PSNR/SSIM on a test split. Locate the real function names with grep and write them in `docs/DESIGN.md` before editing.

## 5. Phases, tasks and acceptance checks

### Phase 0: Setup and baseline
**Already done by the human:** fork cloned with `--recursive`, build succeeded with GCC 14, `fox` sample runs.
**Remaining tasks:** read the Instant-NGP source and write `docs/DESIGN.md` (real function names); write `docs/ENVIRONMENT.md` from the facts above plus whatever I paste from `nvidia-smi`; give me commands to run headless via `scripts/run.py` and to build COLMAP poses for one of our own photo sets with `scripts/colmap2nerf.py`; give me a small baseline benchmark script (render ms/frame, dense mesh extraction time, peak GPU memory) that I run and paste back; save results under `results/baseline_*.csv`.
Accept when: the sample trains and renders; a baseline mesh is saved; baseline timings (render ms/frame, marching steps per ray if measurable, dense MC time, peak GPU memory) are in `results/baseline_*.csv`.

### Phase 1: GPU distance field
Tasks:
- CPU brute-force reference EDT on a small random grid (`tests/`).
- **Exact separable EDT** (Felzenszwalb & Huttenlocher lower-envelope on squared distances): three axis passes, one thread per voxel line, shared-memory tiles. Works on the squared distance, take sqrt at the end.
- **Jump flooding** (Rong & Tan): one thread per voxel, log2(N) passes (7 for N=128), stores nearest-seed coordinates.
- Run per cascade; convert voxel distance to world units using the cascade's scale. Recompute whenever the occupancy grid changes.
- Compare exact vs JFA: runtime, and error vs the CPU reference.
Important design point: **JFA can pick a wrong nearest seed, which makes the distance larger than the truth. A too-large distance is unsafe for skipping because a ray could jump over occupied space.** Use the exact transform for skipping by default; for JFA either apply a conservative safety margin or report it as comparison only. Document which.
Accept when: exact EDT equals the CPU reference exactly (within float epsilon) on several grids; JFA error statistics are recorded; both timed with CUDA events.

### Phase 2: Ray skipping in the marcher
Tasks: add a flag; when a sample lands in empty space, advance by a safe amount derived from the distance field (distance in world units minus a half-voxel-diagonal margin, never smaller than the baseline step); keep the baseline path intact; add a counter for marching steps per ray (atomic or per-ray buffer, off in normal runs).
Accept when: on 2+ scenes, images rendered with skipping match baseline renders (report PSNR/SSIM between the two, target effectively identical) and PSNR/SSIM against ground-truth test views is unchanged; steps per ray drops; ms/frame measured. If skipping ever changes the image, that is a bug in the safe-step bound, fix it first.

### Phase 3: Sparse marching cubes
Tasks:
- Partition the volume into blocks (start with 8³ or 16³ cells; make it a parameter).
- **Block cull**: one thread per block; first use the occupancy grid as a cheap conservative test, then a max-density reduction over the block (with the +1 halo) against the iso level. Compact the active-block list with a prefix sum.
- **Classify**: one thread per cell of active blocks, compute the case index, store triangle count per cell. Case tables in constant memory.
- **Scan**: exclusive scan (thrust or CUB) of triangle counts to get write offsets, so no atomics on the output.
- **Emit**: second pass writes triangles at the offsets. Optional vertex welding.
- Vertex normals; export **GLB** (minimal writer or tinygltf).
- Compare against dense extraction at the same iso level and resolution: extraction time, peak GPU memory, triangle count, **Chamfer distance** (sample ~100k points per mesh) between sparse and dense meshes.
Accept when: sparse mesh agrees with dense (Chamfer reported, near zero); time and memory measured at 256/512/768/1024; GLB opens in the three.js viewer or Blender.

### Phase 4: Service and viewer
Tasks: FastAPI endpoints (upload photos, job status, download results); worker runs COLMAP then headless Instant-NGP then sparse mesh export; scheduler with FIFO and shortest-job-first (job size estimated from image count and resolution, pluggable policy interface); GPU telemetry via NVML (utilisation, memory, power) exposed to the front end; three.js viewer (GLTFLoader, OrbitControls, shaded + wireframe toggle) for the mesh and a NeRF view (pre-rendered turntable video or frames from headless render).
Accept when: end-to-end run from photos to a downloadable GLB; a concurrent-jobs experiment (for example 4 jobs submitted together) compares FIFO vs SJF on turnaround time and GPU utilisation, saved to `results/scheduling_*.csv`.

### Phase 5: Evaluation and writing support
Tasks: Nsight Systems and Nsight Compute profile for each new kernel (occupancy, memory throughput, achieved bandwidth); results tables and plots from the CSVs; a `docs/RESULTS.md` that only quotes measured numbers.

## 6. Benchmark protocol

- Same scene, same viewpoint set, same resolution for baseline and new path. Fixed seeds.
- Warm up (discard first runs), then at least 10 timed runs; report median and spread. Time GPU work with `cudaEvent`, not wall clock.
- Memory: peak via `cudaMemGetInfo` deltas or NVML sampling; state which one.
- Output CSV columns: `experiment, scene, resolution, variant, run, metric, value, unit`.
- Record GPU clocks/thermal state when possible (laptop GPU throttles; keep the machine plugged in and cool).

## 7. Definition of done

All five phases' acceptance checks pass, `docs/PROGRESS.md` lists what was done and why, `docs/DESIGN.md` explains each kernel's thread mapping, `results/` holds raw data for every claim, and the repository builds from a clean clone using the steps in `docs/ENVIRONMENT.md`.

## 8. How to report back after each phase

Reply with: (1) what changed (files), (2) the exact commands I should run to build and test it (you do not run them), (3) what output means pass vs fail, (4) problems or risky assumptions, (5) the proposed next step. After I paste the results back, record the measured numbers in `results/` with their source and update `docs/PROGRESS.md`. Then wait for approval to continue.
