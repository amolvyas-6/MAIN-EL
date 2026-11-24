# Measured results

Measurements were taken on the RTX 3070 Ti Laptop GPU using the headless CUDA
13.5 build. Unless noted, times are medians of ten CUDA-event runs after three warm-up runs.
VRAM numbers are device-wide NVML peaks sampled every 10 ms. See
[Environment](ENVIRONMENT.md) for the toolchain and
[Demo commands](DEMO.md) for reproduction.

## Distance transform

The exact CUDA transform matched the CPU or analytic reference on all eight
fixtures. On the 128³ single-seed fixture, exact EDT took 0.825 ms and jump
flooding took 2.225 ms. Jump flooding found the exact distances on this
fixture. On the random 24³ fixture, it overestimated distances at 13 voxels
with a maximum overestimate of 0.764 voxels. Rendering uses the exact
transform. [Raw distance-transform measurements](../results/distance_transform.csv)

## Ray skipping

The exact-skip renderer matched voxel stepping on the same fixed sample
lattice pixel-for-pixel across all tested views. Compared with the original
renderer, which repeatedly converts positions into stepping space, median
PSNR was 74.84 dB on fused Fox and 94.69 dB on fused Spheres. Spheres used
held-out camera poses and reference images. Its median held-out PSNR and SSIM were
18.12 dB and 0.6455 for both paths at the shown precision.

| Scene and renderer | Original ms/frame | Fixed-lattice voxel ms/frame | Exact skip ms/frame | Original → exact empty advances per pixel ray |
| --- | ---: | ---: | ---: | ---: |
| Fox, fused | 36.374 | 36.376 | 37.273 | 110.80 → 34.91 |
| Fox, unfused | 56.974 | 56.423 | 55.993 | 111.35 → 35.37 |
| Spheres, fused | 41.246 | 41.220 | 41.484 | 25.04 → 20.34 |

The empty advances count is a median across camera views and includes zero
counts for pixels whose rays miss the volume. GPU time is for one fixed
camera view at 800×600. Exact-skip timing uses the cached field, so building
the EDT is measured in its own experiment. Fused frame time rose on both
scenes. The unfused Fox render was 1.8% faster than the original.
[Fox fused CSV](../results/render_fox_fused.csv),
[Fox unfused CSV](../results/render_fox_unfused.csv),
[Spheres fused CSV](../results/render_spheres_fused.csv)

### No-skipping comparison on battery

The 800×600 unfused Fox run checks four modes on the same saved scene and
camera. The no-skipping mode visits every position on the fixed sample grid
and uses occupancy to decide where to evaluate the network. Each mode ran
12 timed frames after warm-up, with the order rotated between rounds.

| Render mode | Median GPU time per frame | Observed range |
| --- | ---: | ---: |
| No skipping | 406.826 ms | 376.517–869.247 ms |
| Original Instant-NGP | 359.580 ms | 353.883–806.266 ms |
| Voxel skipping on a fixed sample grid | 369.050 ms | 355.464–803.537 ms |
| Distance-guided skipping | 380.925 ms | 349.389–798.884 ms |

The distance-guided median was 1.07× as fast as no skipping and 5.9%
slower than original Instant-NGP in this run. The ranges overlap.
Distance-guided and fixed-grid voxel stepping matched pixel-for-pixel
across the four quality-check views. The no-skipping comparison had a
median PSNR of 87.08 dB and a maximum normalized RGBA error of 0.0685.
[Raw battery comparison](../results/render_fox_unfused_ablation.csv)

### Two-mode ray-skipping benchmark

The focused benchmark timed 12 frames per mode on battery at the same
800×600 Fox view. Distance-guided skipping lowered the median from
371.540 ms to 352.061 ms, a 5.2% reduction. The observed ranges were
364.888–392.294 ms for no skipping and 347.015–368.097 ms for
distance-guided skipping. The median number of empty-space advances
per pixel ray fell from 671.17 to 35.37, a 94.7% reduction across the
quality-check views. [Raw two-mode results](../results/render_fox_unfused_no_skip.csv)

## Dense and sparse marching cubes

Both extractors used the same Fox snapshot, bounding box and iso level.
At 256³ and 512³ the triangle counts matched and the sampled, bidirectional
Chamfer distance was zero. At 512³ sparse extraction was 1.26× as fast
with 53% lower sampled peak VRAM. At 768³ sparse extraction produced a
finite mesh and timed successfully. The dense allocation failed at that
resolution, so a paired geometry comparison was unavailable.

| Grid | Dense time (ms) | Sparse time (ms) | Dense peak (MiB) | Sparse peak (MiB) | Paired triangles |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256³ | 55.034 | 53.899 | 1546 | 1174 | 4,885,439 |
| 512³ | 286.613 | 227.017 | 5162 | 2440 | 30,830,938 |
| 768³ | Out of GPU memory | 548.772 | — | 5372 | — |
| 1024³ | Out of GPU memory | Out of GPU memory | — | — | — |

Elapsed time includes geometry, normals, and vertex colors. The sparse
path defers buffers used for later mesh editing. Disk serialization is
outside the CUDA-event timing. Paired mesh samples use fixed-seed
area-weighted sampling and matched triangles.
[Raw mesh measurements](../results/mesh_fox.csv)

## Capture pipeline and viewer

The smoke test uploaded 24 real Fox photos and COLMAP registered all 24.
The worker exported a 449,844-triangle GLB and six NeRF preview frames.
Three.js parsed the GLB and Firefox loaded the WebGL viewer, wireframe
control, NeRF preview, and Measurements page.
[Raw capture measurements](../results/service_smoke.csv)

## GPU job scheduling

The same four Fox photo subsets were submitted into a paused queue and
released together for each policy. All four jobs completed under both
policies. The photo count in the dispatch order identifies the subset.

| Policy | Dispatch order by photo count | Mean turnaround (s) | Batch makespan (s) | Mean sampled GPU utilization |
| --- | --- | ---: | ---: | ---: |
| FIFO | 24 → 16 → 12 → 20 | 97.58 | 146.26 | 23.5% |
| Shortest job first | 12 → 16 → 20 → 24 | 79.16 | 134.89 | 27.0% |

Shortest job first lowered mean turnaround by 18.9% in this batch.
Turnaround includes pose recovery, training and export. GPU utilization is
sampled once per second over the entire batch, including CPU stages.
[Raw scheduling measurements](../results/scheduling_capture.csv)

## Profiling

Nsight kernel occupancy, throughput, and achieved bandwidth were not
measured. The report uses CUDA events and NVML device telemetry.
