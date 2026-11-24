# NeRF2Mesh-GPU design notes

## Verified baseline source map

### NeRF occupancy grid

`include/neural-graphics-primitives/nerf_device.cuh` defines the occupancy
layout used by this checkout:

- `NERF_GRIDSIZE()` is 128 cells per axis and `NERF_GRID_N_CELLS()` is the
  resulting per-cascade cell count.
- `NERF_CASCADES()` is 8. A cascade is addressed by `grid_mip_offset(mip)`.
- Grid cells are Morton ordered by `cascaded_grid_idx_at`. The float density
  grid is accessed by `cascaded_grid_at`; the packed occupancy grid is
  accessed by `density_grid_occupied_at`.

The persistent data are `m_nerf.density_grid` (float values) and
`m_nerf.density_grid_bitfield` (packed occupancy bits). The update path is
`Testbed::update_density_grid_nerf` in `src/testbed_nerf.cu`. It samples and
evaluates density with `generate_grid_samples_nerf_nonuniform`,
`splat_grid_samples_nerf_max_nearest_neighbor`, and `ema_grid_samples_nerf`,
then calls `Testbed::update_density_grid_mean_and_bitfield`. That latter
function runs `grid_to_bitfield` for the finest level and `bitfield_max_pool`
to build coarser occupancy levels.

For Phase 1, the exact distance transform must be rebuilt after this existing
update path changes the occupancy bitfield. The new transform must preserve
the current Morton/cascade indexing or explicitly translate it at its
boundary.

### Ray marching and existing empty-space skipping

The core baseline helper is
`if_unoccupied_advance_to_next_occupied_voxel` in
`include/neural-graphics-primitives/nerf_device.cuh`. It checks the bitfield,
selects a cascade with `mip_from_pos` (or optionally `mip_from_dt`), climbs to
an empty coarser cascade where possible, and calls `advance_to_next_voxel`.
`advance_to_next_voxel` uses `distance_to_next_voxel` and converts its target
back to the renderer's logarithmic stepping space.

There are two rendering paths that call this helper:

- The JIT-fused kernel `render_nerf` in
  `include/neural-graphics-primitives/fused_kernels/render_nerf.cuh` uses one
  CUDA thread per output pixel. `Testbed::render_nerf` in
  `src/testbed_nerf.cu` constructs and launches it when JIT fusion is enabled
  for shaded rendering.
- The non-fused path begins in
  `Testbed::NerfTracer::init_rays_from_camera`, then repeatedly uses
  `generate_next_nerf_network_inputs`, neural-network inference, and
  `composite_kernel_nerf` from `Testbed::NerfTracer::trace`. Its ray payloads
  are compacted between iterations.

Phase 2 must keep both baseline paths available. A distance-guided step must
be conservative and must not replace the current occupancy test merely
because a distance field is present. The fused and non-fused paths are both
correctness-sensitive call sites.

### Dense marching-cubes baseline

`Testbed::marching_cubes` in `src/testbed_nerf.cu` is the host-level NeRF
mesh-extraction entry point. It first calls `Testbed::get_density_on_grid`,
which launches `generate_grid_samples_nerf_uniform`, evaluates the density
network in batches, and writes a dense float grid through
`grid_samples_half_to_float`. It then calls `marching_cubes_gpu`.

`marching_cubes_gpu` in `src/marching_cubes.cu` is a dense two-pass extractor:

1. `gen_vertices` and `gen_faces` count vertices and triangle indices using
   atomics.
2. The host copies those counters, allocates output buffers, then launches the
   same kernels again to emit vertices and indices.

Both kernels use a `4 x 4 x 4` CUDA thread block. `gen_vertices` allocates a
three-plane integer edge-index workspace and interpolates crossings on the
three positive cell edges. `gen_faces` derives a case mask from the eight
density samples and reads its triangle table from a function-local static
table. `compute_mesh_1ring` subsequently launches `accumulate_1ring` to form
vertex normals and neighbourhood data.

`save_mesh` in the same file writes OBJ or PLY. There is no GLB writer in the
baseline. `Testbed::compute_and_save_marching_cubes_mesh` exposes the dense
path to Python, and `scripts/run.py --save_mesh` invokes that binding.

For Phase 3, the sparse path will remain behind a flag and be compared with
this dense `Testbed::marching_cubes` path at the same resolution and threshold.
The dense path evaluates every requested lattice point, so it is the correct
baseline for time, memory, triangle-count, and mesh-agreement comparisons.

### Headless and dataset interfaces

`scripts/run.py` loads a training scene, optionally loads a snapshot, trains
for `--n_steps`, and supports `--save_snapshot`, `--test_transforms`,
`--screenshot_transforms`, and `--save_mesh`. The Python bindings in
`src/python_api.cu` expose `Testbed::render_to_cpu_rgba` as `Testbed.render`
and expose dense mesh extraction through
`Testbed::compute_and_save_marching_cubes_mesh`.

`scripts/colmap2nerf.py` accepts an image directory through `--images` and,
with `--run_colmap`, runs COLMAP feature extraction, matching, mapping,
bundle adjustment, and text export before writing `transforms.json`. It uses
the `sequential` matcher by default; `exhaustive` is the appropriate explicit
choice for an unordered still-photo capture.

## Phase 0 baseline measurement design

The Phase 0 benchmark harness loads an already-trained snapshot, selects a
fixed training view, discards warm-up iterations, and records at least ten
timed iterations. CUDA events measure elapsed time on the streams used by
the baseline work. Dense extraction time includes the end-to-end `Testbed::marching_cubes` work: dense
density evaluation, vertex/face generation, normals, and vertex colours.

The harness samples peak memory through NVML and preserves every raw run
using the project-wide columns:
`experiment,scene,resolution,variant,run,metric,value,unit`.

## Accelerated paths

### Distance field and empty-space traversal

`distance_transform` converts each Morton-ordered occupancy cascade to a
row-major grid. The exact path runs three separable passes over squared
distances. Each CUDA thread builds one 1D lower envelope using shared-memory
arrays for its candidate sites and integer crossing points. The final voxel
kernel takes a correctly rounded square root. An empty cascade produces
infinite distances.

The jump-flood path stores a nearest-seed voxel index and checks the 3D
neighborhood of each voxel at successive halved strides. Each pass reads the
previous pass and writes to a second buffer. Its runtime and error are
measured against the CPU reference. Only the exact field drives rendering
because an approximate seed can overestimate the clearance.

Each device caches the exact field for the current occupancy generation and
refreshes it on the first accelerated render after an update. The marcher
converts the voxel-unit field using the cascade scale. Its lower bound
subtracts two half diagonals, one for the query point and one for the nearest
occupied point. It also clips the jump at lookup-cascade
boundaries. Jumps advance by whole sample indices on a fixed phase per empty
segment. The fused per-pixel kernel and unfused per-ray kernels use the same
skip helper. The optional per-ray counter counts empty advances. Timing
runs disable instrumentation and warm up the cached field.

### Sparse extraction and export

The sparse path uses the same density mask, iso-level convention, lattice
coordinates, and triangle table as the dense extractor. One thread checks
each tile's coarse occupancy footprint including a one-sample halo. The
footprint conservatively covers every potentially contributing cascade.
A scan compacts candidate tile IDs. The density network evaluates only those
tiles in batches, including halo samples.

One CUDA thread block per candidate tile reduces the above- and below-iso
conditions to keep only tiles that contain both. A scan compacts this list.
One thread per cell classifies its corners using the constant-memory case
table. Exclusive scans over triangle counts and tile-edge crossings give
threads disjoint index and vertex output regions. Adjacent cells share
edge vertices inside a tile. The sparse path evaluates vertex colours in
batches and initializes editable-mesh optimizer buffers on demand. GLB export welds identical vertices across
tiles and rebuilds smooth, area-weighted normals.

### Photo service

FastAPI persists job metadata and normalized photos on disk. A local worker
runs PyCOLMAP camera recovery, headless Instant-NGP training, and an isolated
export process. The scheduling policy orders waiting jobs. The GPU worker
runs one job at a time, so starting another job does not contend for its
VRAM. The shortest-job-first estimate accounts for image pairs, total
pixels, training steps, and mesh resolution. NVML reports device-wide
utilization, VRAM use, and power. Next.js polls the queue and telemetry,
reads only recorded CSV measurements, and embeds the Three.js model viewer.

### Algorithm references

- [NGP-RT occupancy distance grid](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03255.pdf)
- [Felzenszwalb and Huttenlocher, Distance Transforms of Sampled Functions](https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf)
- [Rong and Tan, Jump Flooding in GPU](https://www.comp.nus.edu.sg/~tants/jfa/i3d06-submitted.pdf)
