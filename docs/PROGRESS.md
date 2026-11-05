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

### Pending human-run acceptance evidence

- A successful headless fox training/render and a saved snapshot.
- A saved dense baseline mesh.
- `results/baseline_*.csv` containing at least ten CUDA-event render timings,
  ten CUDA-event dense extraction timings, and NVML memory samples.

### Measurements

Not measured. No build, training, rendering, mesh extraction, benchmark, or
profiling command has been run by the agent.

The human reports having run the Phase 0 headless and benchmark commands, but
their terminal outputs and raw CSV contents have not yet been pasted here.
Their success state and metrics therefore remain unrecorded.

### Phase boundary

Phase 1 has not started. Wait for Phase 0 commands to be run and the results
to be pasted before updating measurements or proceeding.
