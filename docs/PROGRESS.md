# NeRF2Mesh-GPU progress

## Phase 0 — setup and baseline

### Completed repository work

- Added `docs/DESIGN.md` after reading the real occupancy-grid, ray-marching,
  dense-marching-cubes, Python, and COLMAP code paths. It records actual
  function names and identifies the fused and non-fused rendering paths.
- Added a Phase 0-only baseline harness: CUDA-event timing bindings in the
  existing Python extension and `bench/baseline.py` for raw CSV output, dense
  PLY/OBJ export, and NVML memory sampling. This changes no normal rendering
  or extraction path; the new bindings run only when explicitly called.
- Added `docs/ENVIRONMENT.md` with the user-confirmed build configuration.
- Recorded the human-provided live GPU, CUDA, and default-GCC version capture.
- Added `docs/COMMANDS.md` as the durable, phase-by-phase command log for
  future agents and contributors.

### Human-run acceptance evidence

- The fox headless command completed, saved `results/baseline_fox.ingp`, and
  rendered frame `0001.jpg` to
  `results/baseline_fox_render/`.
- The baseline harness completed, saved `results/baseline_fox_256.ply`, and
  wrote `results/baseline_fox_800x600_256.csv`.
- The CSV contains ten raw CUDA-event render timings, ten raw CUDA-event
  dense-extraction timings, and NVML device-memory samples for each series.

### Measured baseline results

Source for every value below:
`results/baseline_fox_800x600_256.csv`. Timing is CUDA-event GPU time; memory
is NVML device-wide memory sampled every 10 ms. Spread is the raw min--max
range across the ten timed runs.

| Metric | Configuration | Median | Spread | Notes |
| --- | --- | ---: | --- | --- |
| Render GPU time | fox, 800 x 600, baseline | 13.2495 ms/frame | 13.2055--13.3018 ms | warm-ups discarded before the CSV's 10 timed runs |
| Peak GPU memory | fox, 800 x 600, baseline | 1219.75 MiB used | +36.00 MiB from series start | NVML device-wide sample peak |
| Dense extraction GPU time | fox, 256 x 256 x 256 | 59.6270 ms | 57.3902--62.3350 ms | Includes density evaluation, dense MC, normals, and vertex colours |
| Peak GPU memory | fox, 256 x 256 x 256, dense MC | 2079.75 MiB used | +860.00 MiB from series start | NVML device-wide sample peak |
| Marching steps per ray | baseline | not measured | not measured | No baseline counter exists; Phase 2 will add one behind a flag |

### Phase 0 acceptance status

Passed. The required sample training/render, dense mesh export, render timing,
dense marching-cubes timing, and peak-memory samples are present. No Phase 1
work has started.

## NeRF2Mesh Studio implementation

- Added an exact separable CUDA EDT and a 3D jump-flood comparison. Both
  have CPU-reference checks and a raw timing and error CSV.
- Added an optional cached distance field for the fused and unfused NeRF
  renderers. A fixed sample lattice makes exact skips reproducible.
- Added conservative occupancy tile culling, halo sampling, scan-based
  triangle and edge emission, and vertex-colored GLB export.
- Added the local FastAPI worker, PyCOLMAP camera solving, FIFO and
  shortest-job-first policies, NVML telemetry, and Next.js/Three.js Studio.
- Built the CUDA extension and Next.js production bundle. Verified Fox and
  Spheres renders, the four mesh resolutions, and an end-to-end photo
  upload in Firefox. See [Results](RESULTS.md) and [Demo commands](DEMO.md).
- Verified the same four capture jobs under FIFO and shortest-job-first.
