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
approximations. Against COLMAP's CPU backend on 20 real GoPro frames:

| | measured |
|---|---|
| keypoint positions shared with VLFeat (2 px) | 0.99978 mean, 0.99933 min |
| feature count | 200 722 vs 200 714 (0.004 % apart) |
| descriptor cosine, orientation-matched rows | 0.999980 mean |
| orientation difference | median 0.0004 deg |
| scale ratio difference | median 1.8e-5 |

The small count difference comes from VLFeat's Newton refinement occasionally
converging two candidates onto the same point (~0.06 % duplicate rows); neither
implementation removes them. Compare by position *and* orientation: 37 % of
positions carry more than one orientation, so pairing rows by position alone
reports spurious orientation errors.

The orientation is stored in VLFeat's convention, like the CPU backend.
SiftGPU's databases store a mirrored angle, so the two GPU backends differ
there by design.

**Octaves.** `CusiftExtraction.num_octaves` (default `-1`) picks the octave
count the way SiftGPU does as COLMAP calls it:
`floor(log2(min(width, height))) - first_octave - 3`. This gives 1080p and 4K
inputs each a full pyramid, where VLFeat's fixed default (`num_octaves = 4`)
stops short on large images. A positive value forces that many octaves; set
it to `SiftExtraction.num_octaves` to compare cusift with VLFeat at VLFeat's
own settings.

**Feature cap.** `max_num_features` is applied the way VLFeat applies it:
levels are kept coarse to fine, and the level that crosses the cap is kept
whole. SiftGPU applies the same rule and then caps again after orientation
assignment, dropping its finest levels. On a 1080p daytime frame at the
default `max_num_features = 8192` that gives VLFeat 19 141, cusift 19 143 and
SiftGPU 9 269 keypoints.

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
off and logs that it did. Only `low` and `medium` therefore run identical
settings for `--feature sift` and `--feature cusift`.

### Descriptor-only extraction

`ExtractCusiftDescriptorsForKeypoints` (`colmap/feature/cusift.h`) describes
caller-supplied keypoints. It covers VLFeat octaves >= 0 and refuses keypoints
that map to octave -1 instead of silently sampling them from the wrong pyramid
level. Octave -1 runs the same descriptor code on the upsampled base, whose
parity is covered by the full-extraction comparison above. Results measured
through this entry point exclude the finest-scale features, so do not quote
them as descriptor fidelity over all features.

## Matcher

`CUSIFT_GEMM` computes the full descriptor similarity matrix with one cuBLAS
SGEMM per pair, then reduces it by rows and by columns, so a single pass gives
both directions of the cross-check. It returns the **same match set as
COLMAP's exact CPU brute-force matcher and as SiftGPU's matcher**. On 14 535
pairs from 171 real images, every pair's match set was byte-identical to
SiftGPU's (5 957 071 matches each).

Exactness needed two corrections:

1. **The ratio test runs in both directions.** COLMAP's CPU matcher applies its
   one-way test to the distance matrix and to its transpose before
   intersecting (`sift.cc`, `FindBestMatchesBruteForce`), and SiftGPU does the
   same. Requiring mutual best matches alone admitted ~2.8x as many matches
   between non-overlapping images (62.0 vs 22.4 per pair).
2. **Rank on the raw inner product and convert with a fixed 512^2.**
   `FindBestMatchesOneWayBruteForce` thresholds `acos(dot / 512^2)`, as SiftGPU
   does. Normalising each descriptor instead re-ranks near-ties.

`FeatureMatcherType::SIFT_BRUTEFORCE` with `use_gpu = 0` is *not* an exact
reference by default: with `cpu_brute_force_matcher = 0` it searches a faiss
IVF index approximately (Jaccard 0.976 against the exact matcher). Set
`cpu_brute_force_matcher = 1` when comparing.

**Speed.** The GEMM writes the n1 x n2 similarity matrix to device memory and
reads it back for each reduction, so the matcher is memory-bound: ~11 ms per
pair at 10 000 x 10 000 descriptors, ~18 % slower per pair than SiftGPU
(11.15 vs 9.47 ms over the 14 535 pairs, RTX 4060). `matcher_fused.cu` holds a
single-pass kernel that keeps the similarities on chip, but it is slower than
the GEMM path as built and COLMAP does not use it.

**Dependency.** The matcher links cuBLAS (`CUDA::cublas`, `PRIVATE`); SiftGPU's
matcher does not.

## Validation

Unit tests (`src/colmap/feature/`, `src/pycolmap/feature/`):

- `cusift_test`: VLFeat parity on a synthetic image, with automatic octaves
  and with VLFeat's own octave count; the octave rule; rejected options;
  descriptors usable by the existing SIFT matchers.
- `cusift_matcher_test`: agreement with COLMAP's CPU brute-force matcher,
  ratio test, cross-check and distance thresholds.
- `cusift_determinism_test`: identical extraction and matching output across
  fresh and reused extractors and matchers.

ETH3D regression (`python/ci/test_regression_eth3d.py`, unchanged: `--quality
medium`, 1 deg / 5 cm thresholds, all images required), RTX 5090, one run
each. Wall time is dominated by the mapper, which varies from run to run, so
the totals do not rank the backends.

| scene | backend | rotation error median / max | centre error median / max | extract | match | total |
|---|---|---|---|---|---|---|
| boulders | SiftGPU | 0.100 / 0.128 deg | 0.18 / 0.50 cm | 5.6 s | 0.3 s | 158 s |
| boulders | cusift | 0.093 / 0.120 deg | 0.17 / 0.87 cm | 5.0 s | 1.0 s | 126 s |
| boulders | CPU SIFT | 0.102 / 0.121 deg | 0.17 / 0.86 cm | 6.4 s | 9.5 s | 137 s |
| door | SiftGPU | 0.118 / 0.123 deg | 0.06 / 0.09 cm | 1.7 s | 0.1 s | 42 s |
| door | cusift | 0.296 / 0.306 deg | 0.06 / 0.09 cm | 1.7 s | 0.2 s | 44 s |
| door | CPU SIFT | 0.542 / 0.551 deg | 0.06 / 0.08 cm | 2.6 s | 1.0 s | 46 s |

All runs pass and register every image (boulders 26/26, door 7/7). On door,
each backend's rotation error is nearly constant across images: a whole-model
offset after alignment, not per-image error.

## Licensing

**MIT plus third-party notices**, not plain MIT:

- `LICENSE.CudaSift`: CudaSift (Mårten Björkman, MIT): CUDA scaffolding,
  memory and pyramid plumbing.
- `LICENSE.VLFeat`: VLFeat (BSD): the reference algorithm, and
  `vl_fast_atan2_f` / `vl_fast_sqrt_f` / `vl_fast_resqrt_f` replicated verbatim
  in CUDA. This notice must ship with any redistribution.

## Provenance and local changes

Vendored from the standalone cusift library: the extraction path
(`cudaImage*`, `cudaSift*`, `cudautils.h`) and the matcher (`matcher.h`,
`matcher_gemm.cu`, `matcher_fused.cu`). The library's brute-force warp matcher,
command-line tools and benchmarks are not vendored. Changes made for COLMAP,
to preserve on any re-sync:

1. `GaussKernelSlot`'s constant-bank bookkeeping in `src/cudaSiftH.cu` is
   `thread_local`, not `static`. COLMAP runs one extraction thread per GPU, and
   `d_AdaptiveGaussKernel` is a per-device `__constant__` symbol, so a
   process-wide map would be racy and wrong across devices.
2. `CMakeLists.txt` is COLMAP's. It keeps `--use_fast_math`, which is part of
   the VLFeat-parity numerics (see the comment there), and links
   `CUDA::cudart` and `CUDA::cublas` `PRIVATE`.
3. `src/cusift_device.{h,cu}` are COLMAP additions. They let
   `colmap/feature/cusift.cc` select a device and initialise the memory pool
   without including `<cuda_runtime.h>`, so `colmap_feature` gains no CUDA
   dependency of its own. `InitDevice(-1)` mirrors `colmap::SetBestCudaDevice`'s
   preference order (highest compute capability, then most multiprocessors);
   keep the two in sync.
4. `ExtractSiftDescriptorsOnly` in `src/cudaSiftH.cu` aborts on a keypoint in
   VLFeat octave -1 instead of clamping it to octave 0 (see
   [Descriptor-only extraction](#descriptor-only-extraction)).
5. Keypoints are sorted per octave and orientation histograms use fixed-point
   accumulation (see [Extractor](#extractor), Determinism).
6. The matcher's COLMAP parity work: `ColTop2` tracks each column's runner-up
   so the reverse ratio test can run; the reduction key is selectable (raw
   inner product, or `2*dot - ||other||^2` for true-distance ranking); L2 and
   angular distance types, `maxDistance` and `sqDescriptorNorm` are applied in
   the kernel in COLMAP's units; `MatchDescriptorsGemmU8` uploads uint8
   descriptors (128 B instead of 512 B) and widens them on the device; and the
   second-best index is tracked for the angular ratio test.

## CUDA architectures

`CMakeLists.txt` sets `CUDA_ARCHITECTURES` on `colmap_cusift` instead of
inheriting COLMAP's default of `native`. `-arch=native` emits machine code for
the build machine's GPU and no PTX, so the library would run only on that
architecture. The pinned default `86;89` gives `sm_86` and `sm_89` machine code
plus PTX that JIT-compiles forward to newer GPUs, and keeps codegen the same
across build machines, which matters because `--use_fast_math` is part of the
numerics. Override with `-DCOLMAP_CUSIFT_CUDA_ARCHITECTURES=...`; PTX only
compiles forward, so pre-Ampere targets need an explicit lower virtual
architecture.
