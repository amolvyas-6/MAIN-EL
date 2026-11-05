# Reproducible commands

This is the durable command log for human-run project steps. Agents do not run
these commands. Add a new section when a phase gains a command or its command
changes; preserve the exact working directory, inputs, and output paths.

## Phase 0 — setup and baseline

### Build the current Phase 0 harness

Working directory: `/home/decipher/Projects/MAIN-EL`

```bash
cd /home/decipher/Projects/MAIN-EL
cmake --build build --config RelWithDebInfo -j 8
```

Use GCC 14 only. If CMake configuration must be regenerated, use the GCC 14
configure command recorded in `docs/ENVIRONMENT.md` before this build command.

### Capture the live environment

Working directory: `/home/decipher/Projects/MAIN-EL`

```bash
cd /home/decipher/Projects/MAIN-EL
nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,pstate --format=csv,noheader
nvcc --version
gcc --version
```

The 2026-11-05 output is recorded in `docs/ENVIRONMENT.md`.

### Train and render the fox sample headlessly

Working directory: `/home/decipher/Projects/MAIN-EL`

```bash
cd /home/decipher/Projects/MAIN-EL
PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/run.py \
  --scene data/nerf/fox \
  --n_steps 35000 \
  --save_snapshot results/baseline_fox.ingp \
  --screenshot_transforms data/nerf/fox/transforms.json \
  --screenshot_frames 0 \
  --screenshot_dir results/baseline_fox_render \
  --screenshot_spp 8
```

Expected artifacts: `results/baseline_fox.ingp` and a rendered PNG in
`results/baseline_fox_render/`.

### Measure the dense baseline

Working directory: `/home/decipher/Projects/MAIN-EL`

```bash
cd /home/decipher/Projects/MAIN-EL
PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}" \
python bench/baseline.py \
  --scene "$PWD/data/nerf/fox" \
  --snapshot "$PWD/results/baseline_fox.ingp" \
  --output "$PWD/results/baseline_fox_800x600_256.csv" \
  --mesh-output "$PWD/results/baseline_fox_256.ply" \
  --scene-name fox \
  --width 800 --height 600 \
  --view 0 \
  --warmup-runs 3 --runs 10 \
  --mesh-res 256 --mesh-thresh 2.5 \
  --gpu-index 0 --nvml-poll-ms 10
```

This writes raw CUDA-event timing rows and NVML device-memory samples to the
CSV, then saves a dense PLY mesh. Keep unrelated GPU work closed because NVML
device-memory values are device-wide.

### Build NeRF poses for an unordered personal photo set

Precondition: place capture photos in
`data/nerf/my_capture/images/`. This command is interactive if prior COLMAP
artifacts exist; review the replacement prompt before answering.

Working directory: `/home/decipher/Projects/MAIN-EL`

```bash
cd /home/decipher/Projects/MAIN-EL
CAPTURE_DIR="$PWD/data/nerf/my_capture"
python scripts/colmap2nerf.py \
  --images "$CAPTURE_DIR/images" \
  --colmap_db "$CAPTURE_DIR/colmap.db" \
  --text "$CAPTURE_DIR/colmap_text" \
  --out "$CAPTURE_DIR/transforms.json" \
  --run_colmap \
  --colmap_matcher exhaustive \
  --aabb_scale 32
```

Expected artifact: `data/nerf/my_capture/transforms.json`. Use
`--colmap_matcher sequential` instead only when images were captured in a
continuous, ordered camera path such as extracted video frames.
