# COLMAP with ROCm/HIP Support

AMD GPU (ROCm) backend for COLMAP patch_match_stereo dense reconstruction, enabling GPU-accelerated MVS on AMD GPUs like Radeon RX 7900 XTX.

## Why This Fork?

COLMAP only supports CUDA for GPU-accelerated dense reconstruction. On AMD GPUs, patch_match_stereo falls back to CPU or simply fails. This fork adds a HIP backend so AMD GPU users can run dense stereo matching on their hardware.

Tested on: Radeon RX 7900 XTX (gfx1100) with ROCm 7.0 on Ubuntu 22.04.

## Key Technical Decisions

### 1. Avoid enable_language(HIP) in CMake 3.28+
CMake 3.28 enable_language(HIP) globally pollutes CXX compile flags with -x hip --offload-arch, causing all C++ files to be compiled as HIP which breaks non-GPU code.

Solution: Use add_custom_command() with hipcc to compile only .hip.cpp files, leaving regular C++ compilation untouched.

### 2. CUDA/HIP Dual Compatibility
cudacc.h and cudacc.cc use preprocessor guards (__HIPCC__ / COLMAP_HIP_ENABLED) to switch between CUDA and HIP types/APIs at compile time. No runtime overhead.

### 3. Static Library Circular Dependencies
colmap_mvs <-> colmap_mvs_cuda creates a circular dependency. Resolved with -Wl,--start-group ... -Wl,--end-group linker flags.

## Build Instructions

### Prerequisites
- ROCm 7.0+ installed at /opt/rocm
- CMake 3.28+
- GCC 13+
- Standard COLMAP dependencies (Ceres, Eigen, Boost, etc.)

### Build

```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=OFF \
  -DHIP_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DHIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -G Ninja

ninja -j$(nproc)
```

Set HIP_ARCHITECTURES to match your GPU:

| GPU | Architecture |
|-----|-------------|
| RX 7900 XTX / XT | gfx1100 |
| RX 7800 XT | gfx1101 |
| RX 7600 | gfx1102 |
| RX 6800 XT | gfx1030 |
| MI300X | gfx942 |

### Run

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 colmap patch_match_stereo \
  --workspace_path /path/to/workspace \
  --workspace_format COLMAP \
  --PatchMatchStereo.gpu_index 0
```

Note: HSA_OVERRIDE_GFX_VERSION=11.0.0 is required for RDNA3 GPUs (7900 XTX/XT).

## Performance

On 109 images (1280x720) with Radeon RX 7900 XTX:
- patch_match_stereo: ~86 minutes (GPU init 0.2s, ~4.4s per iteration)
- stereo_fusion: 12 seconds producing 1,261,145 dense points

CPU-only patch_match_stereo on the same dataset crashed after producing only 3,050 points.

## Changed Files

| File | Change |
|------|--------|
| cmake/FindDependencies.cmake | Add COLMAP_HIP_ENABLED definition |
| CMakeLists.txt | Include HIP libs in export set |
| src/colmap/mvs/CMakeLists.txt | HIP compilation with hipcc |
| src/colmap/util/CMakeLists.txt | HIP support for util_cuda |
| src/colmap/exe/CMakeLists.txt | Link HIP libraries |
| src/colmap/exe/mvs.cc | Allow HIP alongside CUDA |
| src/colmap/mvs/cuda_*.h | CUDA-to-HIP header guards |
| src/colmap/mvs/gpu_mat.h | CUDA-to-HIP API calls |
| src/colmap/mvs/gpu_mat_prng.h | hiprand support |
| src/colmap/mvs/patch_match_cuda.h | CUDA-to-HIP guard |
| src/colmap/util/cuda.cc | HIP device management |
| src/colmap/util/cudacc.h | HIP types and safe-call |
| src/colmap/util/cudacc.cc | HIP timer and error handling |

## Known Limitations

- Only patch_match_stereo is GPU-accelerated; SfM feature extraction/matching remains CPU-only
- No SIFT GPU support on HIP (only CUDA)
- Tested on ROCm 7.0 / gfx1100; other GPU archs may need adjustments
- GUI not tested with HIP

## License

Same as COLMAP (BSD 3-Clause).
