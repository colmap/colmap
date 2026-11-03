# cusift (vendored)

GPU SIFT extractor used by `FeatureExtractorType::CUSIFT`
(`src/colmap/feature/cusift.{h,cc}`). It is a **VLFeat replica**, so the right
expectation is that a cusift database matches a COLMAP-VLFeat-CPU database
almost exactly (measured: 0.9998 position overlap, identical feature counts),
and differs from a SiftGPU database by the same margin SiftGPU differs from
VLFeat. It is not, and cannot be, bit-identical to SiftGPU.

Also vendors the GEMM matcher used by `FeatureMatcherType::CUSIFT_GEMM`
(`src/colmap/feature/cusift_matcher.{h,cc}`). Unlike the extractor, the matcher
**is** bit-exact against COLMAP's own reference: measured **Jaccard 1.0000**
(mean, median and min) on 100 real GoPro pairs against both SiftGPU and COLMAP's
exact CPU brute-force matcher, returning an identical 214 714 matches. See
"Matcher: measured status" below for the numbers, including the throughput gap
that is NOT yet closed.

Getting to exactness needed two corrections, which are the substance of this
integration:

1. **The cross-check must apply the ratio test in BOTH directions**, not merely
   confirm mutual-best agreement. COLMAP's CPU path runs its one-way test on the
   distance matrix and again on its transpose before intersecting (`sift.cc`,
   `FindBestMatchesBruteForce`), and SiftGPU applies its thresholds in the column
   direction too. Mutual-best alone admitted ~2.8x too many matches between
   non-overlapping images (62.0 vs 22.4 per pair) and cut same-place/non-overlap
   separation from 190x to 71x.
2. **Rank on the RAW inner product and convert with a FIXED 512^2 constant** --
   not per-row L2 normalization, and not per-descriptor norms.
   `FindBestMatchesOneWayBruteForce` thresholds `acos(dot / 512^2)`, and SiftGPU
   does the same. Normalizing per row re-ranks near-ties and capped agreement at
   Jaccard 0.9898.

**Caution when picking a "reference" matcher.** All four implementations here --
COLMAP's CPU brute force, SiftGPU, this GEMM matcher, and CudaSift's fused
kernel -- are brute force, and all agree bit-for-bit once the convention matches.
The exception is the one that is NOT brute force despite its name:
`FeatureMatcherType::SIFT_BRUTEFORCE` with `use_gpu=0` and COLMAP's default
`cpu_brute_force_matcher=0` builds a faiss `IndexIVFScalarQuantizer` and probes
only 8 of ~400 cells (`feature/index.cc`), i.e. it is **approximate** -- measured
at Jaccard 0.9757 against the exact matcher, the least faithful of the set. Set
`cpu_brute_force_matcher=1` for a true reference; the enum name alone will
mislead you.

## Integration status (measured 2026-07-28, RTX 4060)

`COLMAP_INTEGRATION.md` §5 gate 2, run through this build: 20 frames of the
Menora GoPro 2026_07_16 set at production defaults, the same binary extracting
once with `--FeatureExtraction.type CUSIFT` and once with `type SIFT
--use_gpu 0` (VLFeat CPU).

| | measured | gate |
|---|---|---|
| position overlap vs VLFeat CPU (unique positions, 2 px) | **0.99978** mean, 0.99933 min | ≥ 0.999 ✔ |
| feature count vs VLFeat CPU | 200 722 vs 200 714 rows, **0.004%** apart | see note ✔ |
| descriptor cosine, orientation-matched uint8 DB rows | **0.999980** mean, 99.988% > 0.99 | — |
| orientation | median 0.0004°, 99.993% ≤ 2°; mirror hypothesis rejected at 94° median | — |
| scale | median 1.8e-5, p99 5.0e-4 | — |
| CUSIFT database matched by the existing `SIFT_BRUTEFORCE` matcher | 190/190 pairs verified, 0.9994 inlier ratio | — |
| `sm_86`+`sm_89` cubins **and** PTX, `--use_fast_math` emitted | verified via `cuobjdump --list-elf/--list-ptx` | §4 ✔ |

Two notes on reading those numbers:

- §5's prose asks for *identical* feature counts, which is stricter than
  `HANDOVER.md` §1's own sweep (1 044 483 vs 1 044 653 positions, 0.016% apart;
  §4.2 attributes it to ~0.06% benign Newton-convergence duplicates). The 0.004%
  measured here is well inside that.
- A naive nearest-xy row pairing instead reports 0.976 cosine and a 174°
  orientation p99. That is the metric, not the extractor: 36.9% of rows have a
  co-located multi-orientation sibling, so nearest-xy pairs row A's first
  orientation against row B's second. Group by position and match by orientation
  within it, as `perf_tier1_rtx4060.md` §2c requires, and it is 0.99998.

Extraction throughput in integrated mode, 100 local-disk 1080p frames, identical
`colmap feature_extractor` invocations, 5 interleaved repeats: cusift 41.9 ms/img
median vs SiftGPU 43.4 vs VLFeat CPU 234.9. cusift's own stage timer
(`CUSIFT_TIMING=1`) reports 21.82 ms/img inside COLMAP against 21.1 standalone,
i.e. no integration overhead; extraction is ~52% of pipeline wall time, which
caps the end-to-end gain. Note this rig idles the GPU at 375 MHz and takes a full
pass to reach 2820 MHz — discard the first benchmark pass or you will measure the
clock ramp instead of the extractor.

Not yet measured: the mapper A/B (`HANDOVER.md` §7) — registered images, track
length, reprojection error on a cusift DB vs a SiftGPU DB.

## Matcher: measured status (2026-07-29, RTX 4060)

Identical descriptors throughout (one extracted database, copied per backend),
100 real pairs from the Menora GoPro set: 50 consecutive-frame pairs, and 50
chosen to be >= 8 m apart *with* optical axes diverging >= 93 deg, so
"non-overlapping" means genuinely looking elsewhere rather than merely distant in
frame index.

| | measured | gate |
|---|---|---|
| match-set Jaccard vs SiftGPU, same-place | **1.0000** mean/median/min | = 1.0 OK |
| match-set Jaccard vs SiftGPU, non-overlap | **1.0000** | = 1.0 OK |
| total matches vs SiftGPU | **214 714 vs 214 714** | identical OK |
| match-set Jaccard vs exact CPU brute force | **1.0000** | = 1.0 OK |
| same-place / non-overlap separation | 190.5x raw, 607x verified | matches SiftGPU OK |
| Jaccard under `distance_type=L2` | 0.9895 | reproduces the faiss convention, by design |

Throughput, isolated kernel, same process with the GPU at boost clock (this box
idles at 210-450 MHz of 3105 MHz and cannot lock clocks under WSL, so a
short-lived process measures the ramp, not the kernel), n1 = n2 = 10 000, median
of 30 runs:

| backend | ms/pair | GFLOP/s |
|---|---|---|
| cusift GEMM, float32 upload | 11.98 | 2137 |
| **cusift GEMM, uint8 upload** | **11.00** | **2327** |
| CudaSift fused `FindMaxCorr10`, one direction | 6.51 | 3933 |
| CudaSift fused, both directions | 14.19 | 1804 |

That last row is why this integration keeps the GEMM design rather than adopting
the faster-looking fused kernel: `FindMaxCorr10` writes only `sift1[]`, so a
cross-check costs a second full pass, whereas materialising the similarity tiles
lets one pass serve both directions. For cross-checked matching -- COLMAP's
default -- GEMM wins, 11.0 ms against 14.2 ms.

Inside COLMAP, 100 pairs with geometric verification skipped, 6 interleaved
passes on a load-guarded quiet host (1-minute load 1.45 at start; the harness
refuses to run above 2.0):

| backend | min | median | max | spread |
|---|---|---|---|---|
| `SIFT_BRUTEFORCE` (SiftGPU CUDA) | 5.49 s | **5.92 s** | 6.57 s | 1.20x |
| `CUSIFT_GEMM` | 5.89 s | **6.11 s** | 8.26 s | 1.40x |

**Median ratio 1.031x, min ratio 1.073x: throughput PARITY, not a win.** cusift
neither beats nor meaningfully trails SiftGPU here -- 3% sits at or below the
noise floor when each backend's own spread is 20-40%. One cusift pass was an
8.26 s outlier, so quote medians, not means.

Sizing what remains: the matcher accounts for only ~1.1 s of the ~6 s stage
(100 pairs x 11 ms), so a 3% stage difference implies roughly **1.8 ms/pair** of
matcher-time gap. A pinned host staging buffer is worth ~0.38 ms/pair of that
(both backends currently use PAGEABLE host memory -- verified, neither pins), so
it closes about a fifth; an integer GEMM would be the larger lever, subject to
the signedness caveat recorded at `WidenU8`.

### Why these numbers are load-guarded

An earlier attempt at this table, taken while a C++ IntelliSense server was
reindexing and builds overlapped, produced medians of 5.60 s vs 7.81 s and the
conclusion "cusift is ~1.4x slower". That was an artifact: the SiftGPU baseline
ALONE swung 5.42 s -> 9.64 s between sessions, so the environmental noise
exceeded the effect being measured, and in one session the two backends
alternated which appeared faster. Both `bench_matchers.cu` and
`time_matchers_in_colmap.sh` now refuse to report timings above a 1-minute load
average of 2.0, name the top CPU consumers when they bail, and re-check load
mid-run (a long run can drift into contention and poison only its tail passes).
Do not remove those guards. Clock pinning, the usual control, is unavailable
here: `nvidia-smi --lock-gpu-clocks` requires root and this is WSL.

Note the buffer pre-sizing from attempt 2 was kept regardless of its
inconclusive timing: `Reserve()` growth calls `cudaFree`, which synchronises the
entire device, and in a pipelined host that stalls unrelated queued work. Sizing
once from `max_num_matches` (already clamped to the cache's largest keypoint
count) is correct on principle, not just for benchmarks.

### Full-scale bit-exactness (14 535 pairs)

Exhaustive matching over the same 171-image database, one backend each, geometric
verification skipped, then every pair's match set compared
(`compare_all_pairs.py`):

| | result |
|---|---|
| pairs compared | 14 535 |
| **byte-identical match sets** | **14 535 (100.0000%)** |
| set-equal but reordered | 0 |
| **genuinely different** | **0** |
| total matches | **5 957 071 vs 5 957 071** (identical) |

The bar was set-equality, since row order is not reproducible run-to-run; the
result exceeds it -- even the row order matched, so no allowance was needed. The
"genuinely different" count is exactly zero, which is the only acceptable value:
bit-exact arithmetic with identical tie-break rules leaves no mechanism for a
single disagreement, so one differing pair would have indicated a tie-break
asymmetry or a race, not rounding noise.

This also settles the per-pair throughput question, because 14 535 pairs amortise
the fixed per-invocation overhead that made the 100-pair stage timing above look
like parity:

| backend | total | per pair |
|---|---|---|
| `SIFT_BRUTEFORCE` (SiftGPU) | 137.6 s | **9.47 ms** |
| `CUSIFT_GEMM` | 162.0 s | **11.15 ms** |

**cusift is ~18% slower per pair (1.68 ms).** The in-COLMAP figure of 11.15 ms
agrees with the isolated benchmark's 11.00 ms, cross-validating both harnesses,
and 1.68 ms matches the gap independently inferred from the 100-pair stage times.
Treat 1.18x as approximate -- one unguarded run per backend -- but two
independent methods now agree, so the earlier "parity" reading was an artifact of
fixed overhead dilution, just as the "1.4x slower" reading was an artifact of
competing CPU load. Closing it: a pinned staging buffer is worth ~0.38 ms/pair
(~23% of the gap, landing near 10.8 ms), and an integer GEMM is the larger lever.

### On the mapper A/B, and why it is one cell rather than a 2x2

The original plan called for a 2x2 reconstruction matrix (stock/cusift extractor
x stock/cusift matcher). Bit-exactness makes most of that redundant, and it is
worth spelling out why this is a justified scope reduction rather than a skipped
gate:

- Row ORDER in the matches table is not reproducible even between two runs of the
  same binary, so the real correctness invariant is match-set equality, not byte
  equality. Both consumers are order-invariant: the mapper builds a
  correspondence graph from the set, and two-view geometry estimation does not
  depend on row order.
- If every pair's match SET is equal, then both matchers hand the mapper
  identical input, and the matcher axis of the matrix is settled at the database
  level. That is *stronger* evidence than comparing reconstruction statistics,
  because the mapper is itself nondeterministic (RANSAC): running it would add
  noise on top of provably identical input and could only obscure the result.
- The remaining value of a reconstruction run is as an integration smoke test --
  does the whole pipeline still complete and produce a sane model with the new
  backend wired in -- which one cell answers as well as four.

So: prove set-equality across every pair at scale (not a sample), then run a
single cusift-extractor x cusift-matcher reconstruction against the stock
baseline to close the loop. The set-equality bar admits no partial credit: the
count of pairs whose match sets genuinely differ must be exactly zero, since
bit-exact arithmetic with identical tie-break rules leaves no mechanism for even
one disagreement. A single differing pair means a tie-break asymmetry or a race
and must be understood before trusting the reconstruction.

Result, 669 images, exhaustive matching, `global_mapper` with intrinsics locked:

| cell (extractor x matcher) | registered | points3D | track | reproj |
|---|---|---|---|---|
| SIFT x SiftGPU (baseline) | 669/669 | 335 693 | 10.851 | 0.611 px |
| CUSIFT x SiftGPU | 669/669 | 406 534 | 10.498 | 0.609 px |
| **CUSIFT x CUSIFT_GEMM** | **669/669** | 400 511 | 10.627 | **0.588 px** |
| SIFT x CUSIFT_GEMM | not run -- match sets provably identical to row 1 | | | |

All 669 images register in every cell, and the new path has the lowest mean
reprojection error of the three. The 1.5% points3D difference against
CUSIFT x SiftGPU is pipeline nondeterminism (RANSAC in geometric verification,
plus the mapper itself) on top of provably identical matches -- exactly the noise
the scope-reduction argument above predicts, which is why the database-level
comparison is the stronger evidence and this run is only a smoke test.

## Licensing

**MIT plus third-party notices** — not plain MIT:

- `LICENSE.CudaSift` — CudaSift (Mårten Björkman, MIT): CUDA scaffolding,
  memory/pyramid plumbing.
- `LICENSE.VLFeat` — VLFeat (BSD): the reference algorithm, and
  `vl_fast_atan2_f` / `vl_fast_sqrt_f` / `vl_fast_resqrt_f` replicated verbatim
  in CUDA. This notice must ship with any redistribution.

## Provenance and local changes

Upstream: `exovision-research/cusift`. The extraction path and the GEMM
matcher (`matcher.h`/`matcher_gemm.cu`) are vendored; `matcher.cu` (the
brute-force warp matcher), the standalone CLIs, bench harnesses, and
`stb_image.h` are not imported, because no COLMAP backend targets the
brute-force path and the rest is upstream tooling, not library code.
`matcher.h`/`matcher_gemm.cu` are vendored byte-for-byte unmodified — no local
delta, confirmed by diff against upstream at import time.

Deltas from upstream, all of which must be preserved on any re-sync:

1. `GaussKernelSlot`'s constant-bank bookkeeping in `src/cudaSiftH.cu` is
   `thread_local` rather than plain `static`. COLMAP creates one extraction
   thread per GPU index, which makes a process-wide map both racy and wrong
   across devices (`d_AdaptiveGaussKernel` is a per-device `__constant__`
   symbol). See the comment at the function.
2. `CMakeLists.txt` is COLMAP's, not upstream's. `--use_fast_math` is carried
   over deliberately — see the comment there. `CUDA::cublas` is linked
   `PRIVATE` alongside `CUDA::cudart` for the matcher's cuBLAS SGEMM calls —
   the first cuBLAS consumer in this repo.
3. `src/cusift_device.{h,cu}` are COLMAP additions, not upstream files. They
   exist so `colmap/feature/cusift.cc` can select a device and initialize the
   memory pool without including `<cuda_runtime.h>`, keeping every CUDA type and
   API reference inside this library. `InitDevice(-1)` mirrors
   `colmap::SetBestCudaDevice`'s preference order (highest compute capability,
   then most multiprocessors); keep the two in sync.

## CUDA architectures are pinned per-target

`CMakeLists.txt` sets `CUDA_ARCHITECTURES` on `colmap_cusift` instead of
inheriting COLMAP's project-wide default of `native`. `-arch=native` emits SASS
for the build machine's GPU and **no PTX**, so the result runs only on that exact
architecture and a container image silently inherits whatever card built it.
Verified with `cuobjdump`: `native` gave one `sm_89` cubin and zero PTX, while
the pinned `86;89` gives `sm_86` + `sm_89` cubins plus forward-JIT PTX. It also
keeps codegen reproducible across machines, which matters because
`--use_fast_math` is part of the numerics contract. Override with
`-DCOLMAP_CUSIFT_CUDA_ARCHITECTURES=...` (PTX only JITs forward, so pre-Ampere
targets need an explicit lower virtual arch).

5. `src/matcher_gemm.cu` and `src/matcher.h` carry the matcher-parity work, all
   of which is required for bit-exact agreement with COLMAP's reference matcher
   and none of which exists upstream:
   - `ColTop1` became `ColTop2`, tracking the per-column runner-up so the
     cross-check can apply the ratio test in the reverse direction. Upstream
     only confirmed mutual-best, which is materially weaker on repetitive
     structure.
   - The reduction key is selectable: raw inner product (COLMAP's reference and
     SiftGPU) or `2*dot - ||other||^2` for true-distance ranking (COLMAP's
     faiss path). Row and column reductions share the pattern.
   - `distanceType` (L2 / angular), plus `maxDistance` and `sqDescriptorNorm`
     parameters, so the thresholds are applied in the kernel in the same space
     and units COLMAP uses instead of being approximated by the caller.
   - `MatchDescriptorsGemmU8`, a uint8 entry point that uploads 128 B per
     descriptor instead of 512 B and widens on the device. Its buffer layout is
     deliberately `cublasGemmEx`-compatible; see the signedness caveat recorded
     at `WidenU8` before attempting an integer GEMM.
   - Second-best INDEX tracking (`idx2`), needed by the angular ratio test.

4. `ExtractSiftDescriptorsOnly` in `src/cudaSiftH.cu` now aborts if an input
   keypoint maps to VLFeat octave -1, instead of letting `AssignOctaveLevel`
   clamp it to octave 0. The function is documented "non-upsampled operating
   point only"; clamping was safe when its caller generated the keypoints, but
   `ExtractCusiftDescriptorsForKeypoints` feeds it externally supplied ones,
   where a clamp would produce a descriptor sampled from the wrong pyramid level
   -- silently wrong rather than loudly refused.

## Descriptor-only fidelity: what it measures and what it inherits

`ExtractCusiftDescriptorsForKeypoints` (declared in
`colmap/feature/cusift.h`) describes caller-supplied keypoints, isolating the
descriptor kernel from detection and orientation. Its coverage is deliberately
partial, and the evidence splits in two:

- **Measured directly**: descriptor fidelity for keypoints in VLFeat octave
  >= 0, on identical `(x, y, sigma, theta)` inputs, compared as uint8 rows
  produced by the same COLMAP quantizer at the same normalization.
- **Inherited, not measured here**: octave -1. The upsampled octave runs the
  *identical* descriptor sampling code on a different pyramid base, and that base
  carries its own VLFeat-parity evidence from the original stage gates
  (`HANDOVER.md` §4.2). So octave -1 fidelity follows from the kernel result at
  o >= 0 plus the existing pyramid gates -- it is not a gap in the argument, but
  it is also not something this harness demonstrates.

One selection-bias caveat when quoting numbers from it: octave -1 holds the
finest-scale features, so the excluded set is not random and the measured
population skews larger-scale. Do not quote the result as "descriptor fidelity
over all features", and if it is ever compared against the full-population
conditioned figures (which do include octave -1 pairs), that population
difference is the first thing to check for any gap.

## Why CUDA stays inside this library

`colmap_cusift` links `CUDA::cudart` **PRIVATE** and `cusift.cc` makes no CUDA
API calls of its own. This is dependency hygiene: `colmap_feature` gains no CUDA
runtime dependency of its own and does not need `colmap/util/cuda.h`.

It is **not** a correctness workaround. An earlier revision had `cusift.cc` call
`colmap::SetBestCudaDevice`, which required adding `colmap_util_cuda` to
`colmap_feature`, and that was briefly believed to cause four unrelated test
failures. It does not: a fresh single-shot build of that exact configuration
passes the full suite. Two mechanisms were proposed and both measured false —
`libcudart` NEEDED status is byte-identical either way (upstream already loads it
via `colmap_sift_gpu`), and MXCSR reads `0x1fa0` in both, so FTZ/DAZ are clear.
Inserting `libcolmap_util_cuda.a` into an affected test's link line produces a
**byte-identical binary**, so no symbol provider can have moved either.

## Verify from a fresh build directory

The real cause of those four failures was reused build directories whose
configuration changed mid-flight. Incremental rebuilds after a config change
produced reproducible-looking but false failures in
`controllers/automatic_reconstruction_test`,
`estimators/two_view_geometry_test`,
`estimators_solvers/generalized_relative_pose_test`, and
`scene/reconstruction_pruning_test` — and cost several rounds of misdirected
bisection, including one experiment whose "confirmation" was itself an artifact.

If a change here appears to affect unrelated tests, reconfigure a **fresh** build
directory and build once before believing it.

`estimators/covariance_test` fails independently of this work — it fails on
pristine upstream `c76b19da` in a fresh build on this machine, and is suspected
to be a local Ceres/cuDSS version issue.
