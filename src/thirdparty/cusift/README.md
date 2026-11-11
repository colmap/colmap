# cusift (vendored)

A CUDA SIFT extractor and a CUDA brute-force SIFT matcher, available as two
opt-in backends:

- `FeatureExtractorType::CUSIFT` (`src/colmap/feature/cusift.{h,cc}`)
- `FeatureMatcherType::CUSIFT_GEMM` (`src/colmap/feature/cusift_matcher.{h,cc}`)

They exist for two reasons:

1. **Licensing.** SiftGPU's licence does not permit commercial use. cusift is
   MIT plus BSD third-party notices (see [Licensing](#licensing)), so it gives
   GPU SIFT to users who cannot ship SiftGPU.
2. **Determinism.** For a given image, options and build, extraction and
   matching return byte-identical results on every run
   (`cusift_determinism_test`).

The defaults are unchanged: SIFT extraction and matching still use SiftGPU on
the GPU and VLFeat on the CPU unless cusift is selected.

## Usage

    colmap feature_extractor --FeatureExtraction.type CUSIFT ...
    colmap exhaustive_matcher --FeatureMatching.type CUSIFT_GEMM ...
    colmap automatic_reconstructor --feature cusift ...

The two backends are independent: cusift descriptors are ordinary COLMAP SIFT
descriptors, and `CUSIFT_GEMM` matches descriptors from any SIFT extractor.

## Extractor

**A VLFeat replica, not a SiftGPU replica.** cusift reimplements VLFeat's
detector, orientation assignment and descriptor (the algorithm behind COLMAP's
CPU SIFT backend) on the GPU, including VLFeat's fast `atan2`/`sqrt`
approximations. Its output tracks the CPU backend closely in keypoint
positions, scales, orientations and descriptors. Compare by position *and*
orientation: many positions carry more than one orientation, so pairing rows by
position alone reports spurious orientation errors.

The orientation is stored in VLFeat's convention, like the CPU backend.
SiftGPU's databases store a mirrored angle, so the two GPU backends differ
there by design.

**Octaves.** `CusiftExtraction.num_octaves` defaults to `4`, VLFeat's
`SiftExtraction.num_octaves`, so cusift builds the CPU backend's pyramid. The
count matters beyond the coarse scales: `max_num_features` walks levels from
the coarsest, so extra octaves can push the finest level out of the budget.
`-1` picks the count the way SiftGPU does as COLMAP calls it:
`floor(log2(min(width, height))) - first_octave - 3`.

**Feature cap.** `max_num_features` is applied the way VLFeat applies it:
levels are kept coarse to fine, and the level that crosses the cap is kept
whole. SiftGPU applies the same rule and then caps again after orientation
assignment, dropping its finest levels, so it can return far fewer keypoints
at the same cap.

**Determinism.** The GPU appends keypoints with atomics, so their order
depends on scheduling; cusift sorts each octave's keypoints on the host before
orientation and description. Orientation histograms accumulate in 64-bit fixed
point (2^-32 resolution) instead of float atomics, whose sum depends on the
order of addition.

**Unsupported options are rejected, not ignored.** cusift supports
`first_octave` 0 or -1 and `octave_resolution` 3, and throws on
`estimate_affine_shape`, `domain_size_pooling`, `upright`,
`darkness_adaptivity` and `force_covariant_extractor`. The `high` and
`extreme` presets of `automatic_reconstructor` turn on affine shape, DSP and
guided matching, so with `--feature cusift` the reconstructor switches those
off. Only `low` and `medium` therefore run identical settings for
`--feature sift` and `--feature cusift`.

## Matcher

`CUSIFT_GEMM` computes the full descriptor similarity matrix with one cuBLAS
SGEMM per pair, then reduces it by rows and by columns, so a single pass gives
both directions of the cross-check. By default (`distance_type L2`) it applies
the ratio test of COLMAP's default CPU matcher (`FindBestMatchesIndex`), with
exact search where the CPU matcher's faiss index is approximate. With
`distance_type ANGULAR` it returns the **same match set as COLMAP's exact CPU
brute-force matcher** (`cpu_brute_force_matcher`) while fewer than
`max_num_matches` matches are found (the CPU matcher has no cap), and follows
SiftGPU's matching rules. SiftGPU's match set can still differ: it
keeps only the first `max_num_matches` descriptors of each image, whereas
cusift matches them all and caps the matches.

- **The ratio test runs in both directions.** COLMAP's CPU matcher applies its
  one-way test to the distance matrix and to its transpose before intersecting
  (`sift.cc`, `FindBestMatchesBruteForce`), and SiftGPU does the same.
- **Under ANGULAR, candidates are ranked on the raw inner
  product and converted with a fixed 512^2.** `FindBestMatchesOneWayBruteForce` thresholds `acos(dot / 512^2)`, as
  SiftGPU does; normalising each descriptor instead would re-rank near-ties.

`FeatureMatcherType::SIFT_BRUTEFORCE` with `use_gpu = 0` is *not* an exact
reference by default: with `cpu_brute_force_matcher = 0` it searches a faiss
IVF index approximately. Set `cpu_brute_force_matcher = 1` when comparing.

**Speed.** The GEMM writes the n1 x n2 similarity matrix to device memory and
reads it back for each reduction, so the matcher is memory-bound and somewhat
slower per pair than SiftGPU's.

**Dependency.** The matcher links cuBLAS (`CUDA::cublas`, `PRIVATE`); SiftGPU's
matcher does not.

## Tests

In `src/colmap/feature/` and `src/pycolmap/feature/`:

- `cusift_test`: VLFeat parity on a synthetic image, at both backends'
  defaults (also with the feature cap binding) and with automatic octaves; the
  octave rule; rejected options;
  descriptors usable by the existing SIFT matchers.
- `cusift_matcher_test`: agreement with COLMAP's CPU brute-force matcher
  (ANGULAR) and, at the defaults, with its default index matcher below the
  512 descriptors where that index turns approximate; ratio test,
  cross-check and distance thresholds.
- `cusift_determinism_test`: identical extraction and matching output across
  fresh and reused extractors and matchers.
- `cusift_memory_test`: device memory is released when extractors, matchers
  and their threads go away.
- `cusift_pool_fallback_test`: on a device that cannot create cusift's own
  memory pool, cusift leaves the shared default pool's memory alone.

## Licensing

cusift is MIT (`LICENSE`, Copyright (c) 2026 Yonatan Simson), with two
third-party notices that must ship with any redistribution:

- `LICENSE.CudaSift`: CudaSift (Mårten Björkman, MIT): CUDA scaffolding,
  memory and pyramid plumbing (`cudaImage*`, `cudaSift*`, `cudautils.h`).
- `LICENSE.VLFeat`: VLFeat (BSD-2-Clause): the reference algorithm, and
  `vl_fast_atan2_f` / `vl_fast_sqrt_f` / `vl_fast_resqrt_f` replicated verbatim
  in CUDA (`cudaSiftD.cu`).

Each file's SPDX line and copyright lines name the notices that apply to it.

## Provenance

The CUDA scaffolding (`cudaImage*`, `cudautils.h`, the structure of
`cudaSift*`) comes from CudaSift (https://github.com/Celebrandil/CudaSift),
branch `AdaLovelace` at `ed5ef54c67354959e3d5fa829d30de5b278b4d62`
(2026-03-17).
Detection, orientation and description were rewritten as VLFeat-parity
kernels; the matcher (`matcher_gemm.{h,cu}`) and the device shim
(`cusift_device.{h,cu}`) are new. Notes for anyone changing the code:

1. `GaussKernelSlot`'s constant-bank bookkeeping in `src/cudaSiftH.cu` is one
   table per device, shared by all threads on it and guarded by a mutex.
   `d_AdaptiveGaussKernel` is a per-device `__constant__` symbol, so a single
   process-wide table is wrong across devices, and a per-thread one lets two
   threads on the same device overwrite each other's blur weights when their
   settings differ.
2. Memory: cusift allocates from its own stream-ordered pool per device
   (`cusiftMemPool`, `src/cusift_device.cu`) rather than the device's default
   pool, whose settings stay untouched; `cusift::ReleaseUnusedMemory()`, called
   when an extractor is destroyed, returns the pool's unused memory to the
   driver; and each extractor's scratch (`SiftScratch`) and each matcher's
   buffers and cuBLAS handle (`GemmMatcher`) are freed when it is destroyed.
   They are not `thread_local`: on Windows every CUDA call made from a
   thread-exit destructor fails, so memory freed there leaked.
3. `CMakeLists.txt` keeps `--use_fast_math`, which is part of the VLFeat-parity
   numerics (see the comment there), and links `CUDA::cudart` and
   `CUDA::cublas` `PRIVATE`.
4. `src/cusift_device.{h,cu}` let `colmap/feature/cusift.cc` initialise and
   release the memory pool and read its usage without including
   `<cuda_runtime.h>`. Device selection is COLMAP's own
   (`colmap::SetBestCudaDevice`), so `gpu_index = -1` picks the same device as
   the rest of COLMAP.

## Build

cusift is built with COLMAP's CUDA support when `CUSIFT_ENABLED` is on (the
default). It needs CUDA >= 11.2 for its stream-ordered memory pool, cuBLAS
for its matcher, CMake >= 3.17, and CUDA (not HIP) support enabled. If any is
missing, it is disabled at configure time, and the rest of COLMAP builds as
before. It builds for COLMAP's `CMAKE_CUDA_ARCHITECTURES`, like the other CUDA
targets.
