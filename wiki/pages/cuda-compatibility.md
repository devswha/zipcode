---
title: CUDA Compatibility
tags: [dependencies, troubleshooting]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# CUDA Compatibility

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA Driver | 525+ | 590+ |
| CUDA Toolkit | 12.8+ | 12.8 |
| GCC/G++ | 12+ | 12 |
| CMake | 3.22+ | 3.28+ |

## Why CUDA 12.8+

- CUDA 11.5 (Ubuntu 22.04 default): too old, latest llama.cpp uses unsupported PTX instructions
- CUDA 12.6: `ptxas fatal: Ptx assembly aborted due to errors` on latest llama.cpp CUDA kernels
- CUDA 12.8: works with latest llama.cpp + gcc-12

## GCC Compatibility Matrix

| CUDA Version | Max GCC | Notes |
|-------------|---------|-------|
| 11.5 | gcc-11 | `-compress-mode=size` not supported |
| 12.6 | gcc-12 | ptxas errors on latest llama.cpp |
| 12.8 | gcc-12 | Works. Must set `-DCMAKE_CUDA_HOST_COMPILER=g++-12` |

## Critical: nvcc Path Priority

Ubuntu may have multiple CUDA installations. The system `nvidia-cuda-toolkit` (CUDA 11.5) installs `nvcc` to `/usr/lib/nvidia-cuda-toolkit/bin/` which can shadow `/usr/local/cuda-12.8/bin/nvcc` in PATH.

**Fix**: Set `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` explicitly in cmake.

## Build Script

`scripts/build_llama_server.sh` auto-detects CUDA via `nvcc`. If only `nvidia-smi` is found (driver without toolkit), it warns and builds CPU-only.

## See Also
- [[cuda-build-errors]]
- [[adr-gpu-offload]]
- [[inference-backends]]
