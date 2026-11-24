# Implementation, research gap, and course rationale

This document explains what was added to Instant-NGP, how the major GPU stages work, the project-scoped research gap and contribution, and why the work belongs in both Parallel Architecture and GPU Programming (PAGP) and Computer Graphics and Virtual Reality (CGVR).

For installation and launch commands, start with the root [README](README.md). For exact function names and verified thread mappings, see [docs/DESIGN.md](docs/DESIGN.md); for measured values and raw-data links, see [docs/RESULTS.md](docs/RESULTS.md).

## 1. System overview

The application has two related paths:

```text
Capture path:
photos → FastAPI upload/queue → COLMAP camera poses → headless Instant-NGP
       → occupancy-guided mesh extraction → GLB + rendered views → Three.js viewer

Renderer and evaluation path:
occupancy grid → exact Euclidean distance field → conservative ray skipping
              → same NeRF image with fewer empty-space sample advances

occupancy grid → conservative block culling → density evaluation on candidates
              → marching-cubes classification → prefix scans → compact mesh → GLB
```

Instant-NGP's multiresolution hash-grid network remains the learned scene representation. The new parts accelerate traversal and mesh extraction and provide a complete local workflow for testing and showing those results. Baseline behavior remains available through opt-in flags in the command-line runner.

## 2. Research gap and scope

### What prior work establishes

- Instant-NGP demonstrates fast NeRF training and rendering with multiresolution hash encoding. Its occupancy bitfield already avoids many empty samples, and its dense marching-cubes path is a useful correctness and performance baseline.
- NGP-RT (ECCV 2024) uses occupancy-distance information as part of a complete real-time NeRF renderer. Its result motivates distance-guided traversal, but the contribution is embedded in a larger renderer and is not a drop-in, isolated comparison inside this Instant-NGP fork.
- Felzenszwalb and Huttenlocher describe an exact linear-time separable Euclidean distance transform. Rong and Tan describe jump flooding (JFA), a GPU-friendly propagation method whose nearest-seed result can be approximate.
- Parallel mesh extraction work such as HistoPyramids establishes hierarchical compaction as a way to allocate variable output without globally contending writes.

### Project-scoped gap

For this course project, the practical gap is an integrated and inspectable evaluation of these ideas in the same Instant-NGP checkout: exact versus approximate occupancy distance fields, a correctness-conscious connection to the actual ray marcher, and occupancy-guided sparse mesh extraction compared with the existing dense extractor. The original work does not claim to invent EDT, JFA, ray marching, marching cubes, prefix scan, or a new NeRF representation. It contributes an implementation and measured comparison that make the algorithms' correctness, trade-offs, thread-level parallelism, and end-to-end use visible in one system.

The capture service closes the demonstration gap: a user can provide photos, follow pose recovery and training through a queue, inspect the resulting mesh and NeRF views, and download a standard GLB instead of needing a separate GUI workflow for each stage.

### Research and algorithm references

- Thomas Müller et al., [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/), SIGGRAPH 2022.
- [NGP-RT: Hitting the Limits of NeRFs](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03255.pdf), ECCV 2024.
- P. Felzenszwalb and D. Huttenlocher, [Distance Transforms of Sampled Functions](https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf).
- C. Rong and T. S. Tan, [Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform](https://www.comp.nus.edu.sg/~tants/jfa/i3d06-submitted.pdf), I3D 2006.
- J. Dyken, G. Ziegler, C. Theobalt, and H.-P. Seidel, [High-Speed Marching Cubes Using HistoPyramids](https://www.mpi-inf.mpg.de/departments/computer-graphics/research/publications/2008/high-speed-marching-cubes-using-histopyramids), Computer Graphics Forum 2008.

## 3. GPU distance transform and ray skipping

### Occupancy input

Instant-NGP updates an occupancy bitfield as training proceeds. In this checkout, the update is performed by `Testbed::update_density_grid_nerf` and `Testbed::update_density_grid_mean_and_bitfield` in `src/testbed_nerf.cu`. The source grid uses Morton order and contains 128³ cells per cascade. The distance kernels consume a row-major representation, so the implementation converts at the boundary and handles cascades explicitly.

### Exact separable Euclidean distance transform

For each voxel, the exact method computes the Euclidean distance to the nearest occupied voxel. It first assigns zero to occupied cells and infinity to empty cells, then applies the one-dimensional squared-distance transform along x, y, and z. The one-dimensional transform builds the lower envelope of parabolas in linear time. A CUDA thread processes one voxel line; shared-memory candidate sites and intersection positions hold the envelope. A final voxel pass takes the square root. This separability gives total work proportional to the number of voxels rather than comparing every voxel with every occupied seed.

The result is exact up to floating-point rounding and is the only distance field used for ray skipping. The cached field is refreshed when the occupancy generation changes.

### Jump flooding comparison

JFA stores the nearest occupied seed coordinate at each voxel. Each pass checks a 3D neighborhood at a power-of-two stride; the stride halves until one. A voxel's thread reads the prior buffer and writes the best candidate to the next buffer. This gives regular, highly parallel passes, but propagation can miss the true nearest seed. If the resulting distance is too large, a ray could skip over occupied space. For that reason, JFA is retained as a runtime/error comparison and does not control rendering.

### Conservative integration with ray marching

The baseline helper `if_unoccupied_advance_to_next_occupied_voxel` and both render paths are documented in `docs/DESIGN.md`. Distance skipping remains behind `--use-distance-skip`. For a query in empty space, the marcher converts voxel distance to the cascade's world scale, subtracts a conservative margin that accounts for the query and nearest occupied voxel extents, clips at cascade lookup boundaries, and advances on the existing sample lattice. The ordinary occupancy check still guards network evaluation. Per-ray counters are optional instrumentation; timing runs use the cached transform and disable that instrumentation.

The correctness argument is geometric: after subtracting the voxel-radius safety bound, the proposed jump remains within empty space. Rendering is compared against baseline and fixed-lattice paths at the same cameras. A visual-speed improvement is not assumed: computing a field and looking it up have costs, and fewer sample advances do not guarantee lower frame time on every workload or GPU.

## 4. Occupancy-guided sparse marching cubes

### Dense reference

The baseline NeRF mesh path is `Testbed::marching_cubes` in `src/testbed_nerf.cu`, followed by `marching_cubes_gpu` in `src/marching_cubes.cu`. It evaluates the density field on the full requested grid and performs dense marching-cubes extraction. This remains available when `--sparse-mesh` is omitted.

### Sparse stages

The sparse path uses the same bounding box, density threshold, lattice convention, and marching-cubes case table as the dense path.

1. **Conservative block culling:** blocks (tile size 8 or 16) test a conservative footprint of the occupancy grid, including a one-sample halo. A candidate list is compacted with a prefix scan. This stage may keep extra blocks but must not discard a block that could contain a surface.
2. **Density evaluation and surface rejection:** evaluate the NeRF density only for surviving blocks and their halo. Reduce the block's above/below-isovalue conditions; blocks without a possible crossing are removed and the survivors are compacted.
3. **Cell classification:** one GPU thread handles each cell in the surviving tiles, forms the 8-bit marching-cubes case index and stores its triangle count. Case data is accessed through constant-memory tables.
4. **Prefix-sum allocation:** exclusive scans turn per-cell triangle counts into unique output offsets. A related scan assigns edge vertices. These offsets give every producer a disjoint range, so triangle output needs no global atomic allocation.
5. **Triangle emission and normals:** a second cell pass writes the table-selected triangles. Shared edge vertices are reused within a tile; GLB export welds identical vertices across tiles and creates smooth, area-weighted normals.
6. **Export:** the mesh is written as binary glTF (`.glb`) for standard tools and the browser's Three.js GLTFLoader.

The scan is the key variable-output parallel pattern: each cell can produce zero to five triangles, but a scan gives it an exact output location before emission. This trades scan work and intermediate storage for avoiding dense evaluation, large dense workspaces, and contended output atomics.

### Quality and memory comparison

Dense and sparse runs use the same snapshot, isovalue and grid. Evaluation records extraction time, sampled peak VRAM, triangle count, and fixed-seed sampled bidirectional Chamfer distance. If dense runs out of memory, that is an observed baseline limit; it is recorded as OOM rather than hidden or replaced with fabricated data.

## 5. Photo-to-mesh service and viewer

### API and worker

`service/main.py` provides FastAPI routes for photo upload, job listing/status, queue configuration, GPU telemetry, and completed artifacts. Upload validation accepts 3–120 JPEG/PNG/WebP photos, limits each file and total upload size, normalizes images, and offers 128³/256³/512³ extraction grids and up to 35,000 training steps.

`service/scheduler.py` persists jobs below `service/jobs/` and supports FIFO and SJF policies. SJF estimates workload from the number of images, pixel count, requested training steps, and mesh resolution. One worker processes one capture at a time on the GPU; the scheduling experiment compares queue order and end-to-end turnaround rather than concurrent CUDA kernel execution. `service/capture.py` runs PyCOLMAP feature extraction, matching and incremental mapping, then calls the headless Instant-NGP scripts, exports a sparse GLB and renders representative training views. Each job retains a `worker.log` for diagnosis.

The service samples GPU utilization, memory and power through NVML when supported. Telemetry is device-wide and sampled periodically; it is context, not a per-kernel profiler. Kernel occupancy and achieved bandwidth from Nsight were not measured in the recorded results.

### Web interface

`service/web/` is a Next.js application. The Studio form uploads photos and displays the job queue and stage progress. The viewer uses Three.js to load GLB, orbit/zoom the model, switch shaded and wireframe modes, and show pre-rendered NeRF views. Measurements reads the project's CSVs through `/api/benchmarks`, groups rows by experiment and source file, and presents only available paired measurements.

The default UI API address is `http://127.0.0.1:8000`. Set `NEXT_PUBLIC_NGP_API` at frontend build time to point the browser to another API host, and configure the API's `NGP_WEB_ORIGINS` to permit the frontend origin. See the root [README](README.md) for platform launch instructions.

## 6. PAGP component justification

| Component | Parallel pattern and learning outcome | Evidence to show |
| --- | --- | --- |
| Exact EDT | Independent voxel lines per thread, shared-memory lower-envelope scratch, three separable passes, memory layout conversion, square root and synchronization between passes. Discuss linear work and launch/memory costs. | Exact CUDA output versus CPU brute-force fixtures; runtime CSV for exact EDT. |
| Jump flooding | One thread per voxel per pass, regular neighborhood loads, ping-pong buffers, log₂-scale pass schedule, approximate numerical results. Discuss parallel regularity versus extra passes and error. | JFA runtime and error statistics against the CPU reference; explain why it is not used for safe stepping. |
| Ray traversal | One thread per pixel/ray in fused rendering and per-ray work in the unfused tracer; divergent ray lengths and dynamic step counts; cached field loads and safe bounds. | Baseline and accelerated image comparison, frame times, and steps/empty advances per ray. |
| Sparse block culling | One worker per block, conservative occupancy tests, reductions over density signs, and compaction. | Active block counts and measured dense/sparse time and memory at multiple resolutions. |
| Marching-cubes classification/emission | One worker per cell, case-table lookup, variable triangle counts, scan-generated disjoint offsets, and separate classify/emit passes. | Same-resolution triangle counts and sampled Chamfer comparison; discuss scan cost and memory. |
| Job scheduling | FIFO and estimated shortest-job-first queue policies share one GPU worker; NVML observes GPU state during CPU and GPU stages. | Dispatch order, queue wait, turnaround, makespan, and sampled utilization CSV. |

These patterns make the work suitable for PAGP because the implementation exposes mapping, memory traffic, synchronization, irregular workloads, work avoidance, reductions, scans, and measurement trade-offs directly in CUDA and the GPU scheduling layer.

## 7. CGVR component justification

| Component | Graphics concept | What to demonstrate |
| --- | --- | --- |
| NeRF reconstruction | Multi-view camera geometry and a learned radiance/density field represent the captured scene. | Upload a set of overlapping views and inspect the rendered training-camera images. |
| Volume rendering | Rays sample the field; density becomes opacity and samples are accumulated front to back with transmittance/alpha compositing. | Compare the rendered views and describe how skipping empty space preserves the sample sequence relevant to the image. |
| Occupancy and distance field | A spatial acceleration structure answers where empty space can be traversed quickly. | Visualize/explain occupied voxels, exact distance clearance and the conservative step bound. |
| Isosurface meshing | Marching cubes interpolates an isosurface from scalar density samples; normals support smooth shading. | Compare dense and sparse output, inspect the shaded surface and switch to wireframe to reveal triangles. |
| Asset interchange and interactive viewing | GLB/glTF carries geometry, normals and standard scene data; Three.js provides browser-side WebGL rendering and camera controls. | Orbit/zoom the mesh, inspect wireframe and download/open the GLB in another viewer. |
| Capture workflow | The service links real photos, camera reconstruction, neural rendering and polygonal output into one graphics pipeline. | Follow a job from upload through camera recovery/training to a viewable mesh and rendered NeRF views. |

This is CGVR work because it covers camera pose recovery, volumetric scene representation and rendering, acceleration structures, surface extraction, shading, model interchange, and interactive 3D viewing.

## 8. Measured outcomes and interpretation

Use [docs/RESULTS.md](docs/RESULTS.md) as the authoritative report and `results/*.csv` as the raw data. The recorded RTX 3070 Ti Laptop results include:

- Exact EDT matched the CPU/analytic reference on the listed fixtures. JFA had measurable error on a random 24³ fixture, so only exact EDT drives skipping.
- The focused battery ray test measured 371.540 ms/frame with no skipping and 352.061 ms/frame with exact distance-guided skipping on its Fox view. The corresponding measured median reduction is 5.2%; observed ranges overlap and should be reported alongside the medians. Empty advances fell by 94.7% across the quality-check views. This result applies to this recorded setup; earlier and fused comparisons show that speed changes with path and workload.
- At 512³, sparse mesh extraction took 227.017 ms versus 286.613 ms dense, with equal triangle counts and zero sampled Chamfer distance. At 768³ the sparse run completed at 548.772 ms while dense extraction ran out of GPU memory. At 1024³ both paths ran out of memory.
- On one four-job capture batch, mean turnaround was 97.58 s under FIFO and 79.16 s under SJF (18.9% lower for that batch). This is not a universal scheduling guarantee.
- The end-to-end smoke test registered 24 of 24 bundled Fox photos and exported a GLB with six preview frames.

These values are measurements, not promises. Battery state, thermal limits, GPU clocks, software versions, and workload can affect results. Nsight occupancy/bandwidth profiling was not measured. Do not generalize one small dataset or imply all render modes are faster.

## 9. Reproduction and presentation

1. Use the root [README](README.md) to build the supported Linux setup and start the API/frontend.
2. In the Studio, submit bundled Fox photos (or a suitable personal capture), then inspect the completed mesh and NeRF views.
3. In Measurements, present paired ray modes, mesh results, EDT accuracy, and FIFO/SJF results. Open details to show the CSV source and comparison conditions.
4. Explain the thread mappings and memory/output patterns using the PAGP table, then connect camera rays, volumetric compositing, isosurfaces and WebGL viewing using the CGVR table.
5. For reproducible command lines and benchmark protocol, follow [docs/DEMO.md](docs/DEMO.md). Keep the raw CSV files with any presentation that quotes results.
