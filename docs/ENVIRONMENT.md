# Environment

## Recorded project facts

The following are user-confirmed working-build facts from Phase 0 setup. They
are configuration facts, not benchmark measurements.

| Item | Value |
| --- | --- |
| Operating system | Arch Linux |
| GPU | NVIDIA RTX 3070 Ti Laptop GPU |
| GPU architecture target | Ampere, compute capability 8.6 (`TCNN_CUDA_ARCHITECTURES=86`) |
| GPU memory | 8 GB reported hardware capacity |
| CUDA toolkit | CUDA 13.4, `nvcc` 13.4.92 at `/opt/cuda` |
| Default compiler | GCC 16.2.1; unsupported for this build because bundled CUTLASS fails to compile |
| Working host compilers | `gcc-14` and `g++-14` |
| Python | Python 3.12 |
| tiny-cuda-nn setting | `TCNN_HALF_PRECISION=ON` |
| Sample status | `./instant-ngp data/nerf/fox` previously ran successfully |

## Build command

Run from `/home/decipher/Projects/MAIN-EL`. Use this exact configure command
when configuration must be regenerated (for example, a clean build directory
or changed CMake configuration):

```bash
cmake -B build -DNGP_BUILD_WITH_GUI=off -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build build --config RelWithDebInfo -j 8
```

Do not use GCC 16 for this project. Keep parallel builds at `-j 8` or lower
because CUDA compilation is memory intensive on this laptop.

## Live version capture

The human ran the following command on 2026-11-05 from the project terminal:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,pstate --format=csv,noheader
nvcc --version
gcc --version
```

Reported output:

```text
NVIDIA GeForce RTX 3070 Ti Laptop GPU, 615.78.08, 8192 MiB, 41, P0
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Sep_01_08:45:21_PDT_2026
Cuda compilation tools, release 13.4, V13.4.92
Build cuda_13.4.r13.4/compiler.38855100_0
gcc (GCC) 16.2.1 20260810
Copyright (C) 2026 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

This confirms NVIDIA driver 615.78.08, 8192 MiB reported GPU memory, CUDA
13.4.92, and default GCC 16.2.1 at capture time. GCC 14 remains mandatory for
this project's CUDA build.

## Measurement constraints

The 8 GB VRAM capacity makes a dense 1024-cubed float field approximately
4.3 GB before temporary and output allocations. Dense high-resolution mesh
tests may therefore run out of memory; that outcome must be recorded as a
result rather than hidden.
