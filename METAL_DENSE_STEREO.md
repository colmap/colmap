# COLMAP dense stereo on Apple Metal

This source tree adds an experimental native Metal compute backend for COLMAP's
PatchMatch dense-stereo stage. It builds as an arm64 executable and preserves
the existing `patch_match_stereo` command, workspace layout, depth maps, normal
maps, selection-probability maps, and consistency-graph formats.

## Build on Apple Silicon

Install COLMAP's dependencies with the Apple Silicon Homebrew installation in
`/opt/homebrew`, then configure a clean build:

```sh
cmake -S . -B build-metal \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_PREFIX_PATH=/opt/homebrew \
  -DCMAKE_IGNORE_PREFIX_PATH=/usr/local \
  -DOpenMP_ROOT=/opt/homebrew/opt/libomp \
  -DCUDA_ENABLED=OFF \
  -DHIP_ENABLED=OFF \
  -DMETAL_ENABLED=ON \
  -DFETCH_POSELIB=ON
cmake --build build-metal --target colmap_main -j
```

`CMAKE_IGNORE_PREFIX_PATH=/usr/local` matters on Macs that retain an Intel
Homebrew installation. It prevents CMake from mixing x86_64 libraries into the
arm64 build.

Run dense stereo with the normal COLMAP interface:

```sh
build-metal/src/colmap/exe/colmap patch_match_stereo \
  --workspace_path /path/to/undistorted/workspace \
  --PatchMatchStereo.gpu_index 0
```

The default GPU index `-1` also selects the system Metal device. Multi-device
execution is not implemented because Apple Silicon presents one unified GPU.

## Implementation

The host layer is Objective-C++ and keeps all Metal framework types behind the
portable `PatchMatch` C++ interface. At runtime it compiles embedded Metal
Shading Language kernels and allocates shared-memory buffers for:

- the reference image and a padded source-image atlas;
- packed intrinsics, relative poses, and projection matrices;
- depth, normal, per-source matching-cost, and persistent PRNG maps;
- source depth maps for geometric consistency;
- current/previous per-source probabilities, scanline workspace, and the final
  consistency mask.

Three compute phases run on the GPU:

1. Random plane initialization and a bilaterally weighted NCC cost volume.
2. Four rotated scanline sweeps per iteration. Each sweep computes backward
   and forward MRF messages, applies viewing-angle and resolution priors,
   Monte Carlo samples source views, and tests the same five propagated/random
   plane candidates as the CUDA implementation.
3. Photometric/geometric filtering and consistency-mask generation from the
   final source-selection probabilities.

The source images remain 8-bit grayscale, matching the current CUDA input. The
result buffers are converted back to COLMAP's slice-major `Mat` representation
before the existing workspace writer receives them.

## Validation

The integrated test constructs a calibrated two-view problem, runs the public
`PatchMatch` wrapper on Metal, and validates every returned depth, normal, and
probability:

```sh
cmake --build build-metal --target colmap_mvs_patch_match_metal_test -j
build-metal/src/colmap/mvs/patch_match_metal_test
```

For a framework-only kernel check that does not link the rest of COLMAP:

```sh
xcrun clang++ -std=c++17 -fobjc-arc \
  -framework Foundation -framework Metal \
  scripts/check_metal_shader.mm -o /tmp/check_colmap_metal_shader
/tmp/check_colmap_metal_shader src/colmap/mvs/patch_match_metal.mm
```

### Real-scene benchmark

The backend has also been exercised on the official 128-image [South Building
dataset](https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip).
At a 320-pixel maximum image dimension, a 15-view subset produced photometric
and filtered geometric maps that COLMAP's existing `stereo_fusion` consumed
without format conversion. Geometric filtering retained 44.7% of pixels. Of
32,954 valid checks against sparse observations, median relative depth error
was 0.35%, the 95th percentile was 1.47%, and 99.2% were within 5%. The fused
result contained 66,862 points. These sparse checks detect integration errors;
they do not replace evaluation against dense ground truth.

Use the sparse reconstruction as a checkpoint (not as dense ground truth) and
summarize the generated maps with:

```sh
python3 scripts/evaluate_metal_dense.py \
  --workspace /path/to/dense/workspace \
  --model-text /path/to/dense/workspace/sparse-text \
  --input-type geometric \
  --ply /path/to/fused.ply \
  --output /tmp/metal-dense-report.json
```

The evaluator reports valid-pixel coverage, sparse-observation depth error,
consistency-graph coverage, and the fused PLY vertex count. Convert a binary
sparse model first with `colmap model_converter --output_type TXT`.

### DTU dense-ground-truth benchmark

The Metal backend has been evaluated on scan 6 from DTU's official SampleSet.
The selective downloader fetches about 186 MiB rather than storing the complete
6.3 GB archive:

```sh
python3 scripts/download_dtu_scan.py \
  --output data/metal-benchmark/dtu \
  --scan 6 --light 3
python3 scripts/prepare_dtu_colmap.py \
  --sample-root data/metal-benchmark/dtu/SampleSet \
  --workspace data/metal-benchmark/dtu/scan6-metal-800 \
  --scan 6 --light 3 --max-image-size 800
```

The preparation script directly decomposes DTU's calibrated projection
matrices, verifies their recomposition, derives the PatchMatch depth range from
the structured-light scan, and ranks source views by frustum overlap and
triangulation angle. It also writes one zero-error connectivity track because
COLMAP's fusion view graph is derived from sparse tracks, not
`patch-match.cfg`; the track is not used to initialize PatchMatch.

For scan 6, the generated depth interval was 477.386434–1051.008168 mm. The
measured run used ten source images, ten samples, three iterations, geometric
consistency, and a three-pixel fusion threshold:

```sh
build-metal/src/colmap/exe/colmap patch_match_stereo \
  --workspace_path data/metal-benchmark/dtu/scan6-metal-800 \
  --PatchMatchStereo.gpu_index 0 \
  --PatchMatchStereo.depth_min 477.386434 \
  --PatchMatchStereo.depth_max 1051.008168 \
  --PatchMatchStereo.num_samples 10 \
  --PatchMatchStereo.num_iterations 3 \
  --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.filter 1 \
  --PatchMatchStereo.filter_min_num_consistent 2 \
  --PatchMatchStereo.write_consistency_graph 1
build-metal/src/colmap/exe/colmap stereo_fusion \
  --workspace_path data/metal-benchmark/dtu/scan6-metal-800 \
  --input_type geometric \
  --StereoFusion.min_num_pixels 3 \
  --output_path data/metal-benchmark/dtu/scan6-metal-800/fused-geometric.ply
python3 scripts/evaluate_dtu_pointcloud.py \
  --reconstruction \
    data/metal-benchmark/dtu/scan6-metal-800/fused-geometric.ply \
  --sample-root data/metal-benchmark/dtu/SampleSet \
  --scan 6 \
  --output data/metal-benchmark/dtu/scan6-metal-800/dtu-evaluation.json
```

With source images stored in an `R8Unorm` Metal texture array, sampled by the
hardware linear sampler, and bilateral spatial/color exponentials replaced by
small constant-cache lookup tables, the two PatchMatch passes over 49 images at
800×600 completed in 32.0 seconds on an Apple M5 Max. The original manual
four-tap buffer sampler took 65.7 seconds, so this is a 2.05× end-to-end
speedup. The optimized run retained 61.709% of geometric depth pixels and
fusion produced 910,816 points. The supplied DTU mask, ground plane, 0.2 mm
reduction, and 20 mm statistics cutoff yielded 0.3572 mm accuracy, 0.4893 mm
completeness, and 0.42323 mm overall mean distance. Accuracy and completeness
were respectively 95.0% and 95.1% within 1 mm. This is a reproducible
single-scan engineering result, not a directly comparable full-DTU leaderboard
submission.

For comparison, the buffer-sampled run retained 61.709% of depth pixels,
produced 910,656 points, and scored 0.42325 mm overall. The texture-only run
took 34.4 seconds and scored 0.42337 mm. The constant-cache lookup version uses
7.1% less time than texture sampling alone and differs from the original score by
only 0.000018 mm (0.004%). The fast-approximation artifacts and evaluator report are under
`data/metal-benchmark/dtu/scan6-metal-lut-800`.

The backend logs command-buffer wall time and Metal's reported GPU time for
each solve. On the exact same representative 800×600 photometric view, the
twelve sweep dispatches fell from 865.9 ms to 251.7 ms and total reported GPU
time fell from 940.5 ms to 277.9 ms. This is a 3.44× sweep speedup and a 3.38×
GPU-time speedup. The sweep still accounts for roughly 91% of GPU time.
Straightforward per-thread reference-stat reuse and paired-source batching were
benchmarked but rejected because their extra register pressure made the kernel
slower. A 32-lane cooperative patch reduction was also rejected after raising
representative sweep GPU time from 245.6 ms to 327.2 ms. Incremental and
CPU-prepacked homography variants were neutral or slower. An 8x8 threadgroup
improved one photometric view to 235.0--235.9 ms, but the complete 49-view run
showed 5.1% higher geometric sweep time than 16x16, so the full-run winner
remains the 16x16 lookup-table implementation reported above.

## CUDA-fidelity sweep

The active backend now implements CUDA's serial top-to-bottom column
dependency rather than the earlier double-buffered approximation. One Metal
SIMD group owns each logical column. Map rotation is represented by coordinate,
normal, and intrinsic transforms, so the depth, normal, cost, probability, and
PRNG state remain in place while producing the same four sweep directions.

The source-selection path includes the CUDA forward/backward MRF, previous-pass
probability blending, triangulation/incident-angle/resolution priors, and Monte
Carlo source sampling. A custom deterministic 32-bit PRNG replaces `curand`.
The SIMD scheduler draws all sources in the original order, distributes the
four candidates times all samples across 32 lanes, evaluates each NCC patch in
the CUDA scalar accumulation order, and sums each candidate in sample order.
This reduced the representative 800x600 view from 8.22 seconds for one scalar
thread per column to about 0.62 seconds total GPU command time; its twelve
sweeps take about 0.59--0.62 seconds on an Apple M5 Max.

A clean 49-view DTU scan-6 run at 800x600, with ten sources, ten samples, three
iterations, and photometric plus geometric passes, completed in 72.3 seconds.
Geometric filtering retained 66.986% of pixels and fusion produced 401,004
points. The DTU evaluator measured 0.3996 mm accuracy, 0.5766 mm completeness,
and 0.48808 mm overall mean distance. Reproducible artifacts are under
`data/metal-benchmark/dtu/scan6-metal-fidelity-exact-800`.

The earlier best-source approximation remains faster (32.0 seconds) and scored
better on this one scan (0.42323 mm overall), so the fidelity port is not an
automatic quality improvement. Different PRNG sequences and floating-point
execution mean Metal and CUDA will not be bit-identical. Production adoption
still needs matched CUDA/Metal comparisons, the full DTU test split, Tanks and
Temples, and testing across multiple Apple GPU generations.
