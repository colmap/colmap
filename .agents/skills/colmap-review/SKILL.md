---
name: colmap-review
description: >-
  Maintainer-grade code review for COLMAP (C++, CUDA/HIP, pycolmap, CMake, and
  benchmarks): checks geometric and numerical correctness, modularity and
  readability, memory/binding safety, performance, and test quality, while
  shrinking bloat, reusing existing helpers, and localizing changes. Use for
  "review this", "code review", "review the last commit", "review my PR", "can
  this diff be smaller", "is this over-engineered", or /colmap-review. Don't use
  when asked to implement changes or run benchmarks directly.
---

Review a COLMAP change like a core maintainer who will own every added line for
years. Order of importance: **correct**, **modular, readable & lean**, **safe**,
**fast**, **tested**, **consistent**. Every unnecessary option, abstraction, or
duplicate branch is a permanent maintenance burden — reduce total diff size and
confine changes to as few locations as possible, while keeping functions short,
self-documenting, and well-scoped rather than cramming logic into monoliths.
This is a report the user asked for, so give it in full. Lists findings, changes
no code.

## 1. Understand first

- **Scope the target:** Review what the user names: uncommitted or staged
  changes, a commit/branch (`git diff`), a GitHub PR (`gh pr view` /
  `gh pr diff`), or specific files. Nothing named: review uncommitted and staged
  changes, or `HEAD` (`git show`) if the worktree is clean.
- **Audit the diff footprint:** Inspect `git diff --stat` up front. Ask why each
  file is touched and whether the same goal can be achieved in fewer locations
  with less new code, respecting COLMAP's bottom-up module layers (`AGENTS.md`).
- **Read connected code, not only the diff:**
  - Grep every caller and consumer across `src/colmap/`, `src/pycolmap/`,
    `python/`, `benchmark/`, and `doc/` whenever a signature, return contract,
    default, enum, database schema, or persisted format changes. Broken or stale
    call sites often sit outside the diff.
  - Search sibling files and lower-level modules (`util/`, `math/`, `geometry/`,
    `sensor/`, `scene/`, `estimators/`) to spot existing helpers, cost wrappers,
    and idioms that new or duplicated code should reuse.
  - Trace data and coordinate frames end-to-end: where poses, intrinsics,
    observations, and IDs enter, what invariants hold, and what mutates in
    place.

## 2. Look for

1. **Modularity, readability & diff size:**
   - **Small functions, tight scopes & clear names:** Prefer short, cohesive
     functions with self-documenting names and tightly scoped (`const`) local
     variables over long functions with wide variable scopes, mutable state
     carried across stages, or dense blocks that are hard to follow without
     comments. Check whether distinct blocks of a long function can be extracted
     into shorter helpers — and check whether those helpers already exist
     elsewhere in the codebase before introducing new ones. Add concise comments
     where non-obvious math, invariants, or trade-offs need context rather than
     leaving opaque blocks uncommented or narrating obvious code.
   - **Delete bloat & localize changes:** Question dead code, unused helpers or
     `#include`s, debug `LOG(INFO)` noise, speculative options/callbacks nobody
     sets, unrelated changes bundled into the PR, and hand-rolled Jacobians or
     special-case fast paths without a measured win over Ceres autodiff or the
     general path. Push scattered `if/else` plumbing across callers behind the
     owning domain type (`Camera`, `Rig`, `Frame`, `TwoViewGeometry`) and define
     option defaults in one place rather than syncing duplicates across
     `OptionManager` and estimator structs.
   - **Reuse COLMAP building blocks & keep designs flat:** Reuse existing cost
     wrappers (`CovarianceWeightedCostFunctor`, `ScaleWeightedCostFunctor`),
     `MAKE_ENUM`/`MAKE_ENUM_CLASS`, `OptionManager`,
     `AddCameraWithTrivialRig`/`AddImageWithTrivialFrame`, `math/` and
     `geometry/` utilities, and Eigen vector/matrix operations instead of
     elementwise loops. Prefer free functions over single-method classes, a
     `struct` over a trivial `class`, return values (`T` or `std::optional<T>`)
     over out-parameters, and flat `*Options` structs (no inheritance). Keep
     headers small by moving non-trivial bodies and internal helpers to
     anonymous namespaces in `.cc`.

2. **Geometric, numerical & SfM/MVS bugs:**
   - **Camera model generality & coordinates:** Assuming a finite perspective
     plane (`CamFromImg`, `z = 1`) where wide-FOV fisheye (`>= 90°`) or
     spherical rays are valid — use unit bearings (`CamRayFromImg` /
     `CamRayFromImgWithJac`). Check continuous pixel-center (`+0.5`) vs.
     integer-grid coordinate offsets across feature extraction, warping, and
     interpolation, and seam wrap-around on panoramic/spherical cameras.
   - **Solvers, residuals & degeneracies:** Division by zero or `.normalized()`
     on zero/degenerate vectors producing `NaN`; `LLT`/`LDLT` without
     positive-definiteness checks (`LDLT::info() == Success` allows indefinite
     or singular matrices); scale-dependent rank/conditioning thresholds;
     degenerate configurations (collinear points, pure rotation / central rigs
     with unobservable scale — including on the final RANSAC inlier subset, not
     only all inputs); solver outputs written back without checking convergence
     or `allFinite()`; small-angle / gimbal singularities. In RANSAC or pose
     scoring, ensure behind-camera or failed projections are penalized rather
     than counted as zero error; in Ceres cost functors, avoid returning `false`
     on single outlier observations when a step should not abort.
   - **Frames, manifolds & determinism:** Missing Ceres manifolds
     (`EigenQuaternionManifold`, `SphereManifold`); wrong `target_from_source`
     (`cam_from_world`, `rig_from_world`, `sensor_from_rig`) composition,
     inverse conjugation, or left- vs. right-perturbation Jacobians; `Rig`
     invariants (`IsRefSensor` identity pose vs. optional `sensor_from_rig` on
     non-reference sensors); nondeterminism from iterating
     `FlatHashMap`/`FlatHashSet` when order affects tie-breaking, gauge fixing,
     or serialized output, or from unseeded/reseeded PRNGs.

3. **Safety, error contracts & cross-language (`pycolmap` / CMake) risks:**
   - **State & database safety:** Validate preconditions *before* mutating
     `Reconstruction` or `Database` so exceptions never leave inconsistent links
     across `Camera`/`Rig`/`Image`/`Frame`/`Point3D`/`Track`. Use
     `THROW_CHECK*` / exceptions rather than `LOG(FATAL)` in library and worker
     code so `pycolmap` and the GUI surface errors cleanly instead of
     terminating the process, while skipping single diverged cameras or points
     in pipelines with `LOG(WARNING)`. For `Database` or format changes, check
     backwards compatibility, transaction boundaries, referential cleanup, and
     stability of persisted enum values.
   - **`pycolmap` bindings:** Dangling pointers/references or NumPy views into
     mutable C++ containers; raw `T*` arguments without `None` handling; C++
     mutating STL containers by pointer across pybind11's copy-converting
     boundary (updates silently lost in Python); inconsistent GIL release on
     heavy work; copy/pickle support (`MakeDataclass`); missing option or
     docstring parity across C++, `OptionManager` CLI, GUI, and `pycolmap`.
   - **Build & CI portability:** Inconsistent feature gates
     (`COLMAP_CUDA_ENABLED`, `COLMAP_HIP_ENABLED`, `COLMAP_GPU_ENABLED`,
     `COLMAP_GUI_ENABLED`, `COLMAP_ONNX_ENABLED`); GPU tests missing runtime
     `GTEST_SKIP()` when no device is available; missing CMake target/config
     exports; Windows/MSVC portability (path handling, narrowing/conversion
     warnings, missing headers); formatting/lint breaks (`scripts/format/c++.sh`,
     `scripts/format/python.sh`).

4. **Performance & algorithmic evidence:**
   - Flag accidental `O(N^2)` scans or insertions, hash lookups or heap
     allocations inside inner loops and `std::sort` comparators, redundant image
     I/O or unbatched SQLite queries, and avoidable hot-loop work (copying Eigen
     matrices/descriptors or loop elements instead of `const auto&`, `.norm()`
     instead of `.squaredNorm()` in distance comparisons, recomputing
     loop-invariant transforms).
   - Algorithmic or default-parameter changes in feature matching, RANSAC, SfM,
     or MVS require empirical justification (accuracy + runtime on benchmarks
     such as ETH3D or multi-seed synthetic sweeps). Push back on unverified
     default changes or inner-loop slowdowns.

5. **Missing tests & test bloat:**
   - **Missing or vacuous tests:** Non-trivial logic, new solvers/cost functors,
     and bug fixes need a focused C++ test (`*_test.cc`) that fails without the
     change. Catch vacuous assertions: checks that pass on empty/degenerate
     reconstructions, bare `EXPECT_ANY_THROW` that could pass on an unrelated
     error, or missing coverage of edge cases and singularities.
   - **Keep tests lean:** Functional tests belong in C++ (`*_test.cc`) — keep
     `pycolmap` tests (`*_test.py`) focused on the binding layer (`None`/empty
     inputs, types, lifetime) unless non-trivial logic lives in Python. Merge
     repetitive cases into `TEST_P` or table-driven loops; reuse
     `ReconstructionEq`, `Rigid3dNear`, `Sim3dNear`, `EigenMatrixNear`,
     `SynthesizeDataset`, and `Database::Open(kInMemorySqliteDatabasePath)`
     instead of verbose field loops, temp directories, or slow end-to-end
     pipeline runs; gate heavy model-inference tests behind `COLMAP_TEST_HEAVY`.

6. **COLMAP conventions (flag concrete deviations, never personal taste):**
   - Follow `AGENTS.md`: `FlatHashMap`/`FlatHashSet` by default (no
     iterator-erase loops), `NodeHashMap`/`NodeHashSet` when pointer/reference
     stability across mutations is needed, never `std::unordered_map/set`, and
     `std::map/set` only when sorted order is required; strong IDs (`camera_t`,
     `image_t`, `frame_t`, `rig_t`, `sensor_t`, `point2D_t`, `point3D_t`);
     `target_from_source` and `x_in_y` naming; output pointer/ref params last;
     strict `const` on locals and immutable members; `//` comments (not `///`);
     `// SPDX-License-Identifier: BSD-3-Clause` on new files.

## 3. Check before you report

- **Concrete case or concrete simplification:** Every bug, safety, or
  performance finding needs a concrete trigger and failure ("when two views
  undergo pure rotation, normalizing the zero baseline vector feeds `NaN` into
  Ceres"). Every modularity, readability, or diff-size finding needs a verified
  simplification ("extract the 40-line ray-angle filter into a helper and reuse
  `CalculateTriangulationAngle` from `geometry/triangulation.h`, scoping locals
  tightly and saving ~25 lines"). No case, no finding.
- **Verify against the repo:** Re-read the lines, callers, and any COLMAP helper
  or matcher you suggest reusing to confirm exact signatures, visibility, and
  module-layer rules. When a `build/` directory or Python environment is
  available and a subtle numerical, SQLite, or formatting question arises,
  verify with a quick read-only check (targeted `ctest` or scratch calculation)
  — never modify or leave untracked files in the workspace.
- **Smallest, cleanest fix wins:** Prefer fixes that reuse existing COLMAP
  primitives, split long functions into small well-scoped helpers, delete dead
  or duplicate code, and touch fewer files. Never propose new classes, layers,
  or options unless collapsing real complexity or duplication.
- **No fluff:** Respect documented limitations and intentional trade-offs unless
  broken under realistic SfM/MVS workloads. No vague "consider", no speculative
  worries, no praising files. Explicitly call out broken callers or stale docs
  outside the diff.

## 4. Output

Use direct, concise technical English: short sentences, exact COLMAP symbol
names, and `path/to/file.cc:L10-20` references. Do not explain standard C++,
Eigen, Ceres, or SfM concepts.

Start with `What this change does:` in two or three sentences, including the
diff footprint (`+A/-D` across `F` files) and whether the overall approach is
appropriately modular and localized.

Then list findings in three groups (skip empty groups), numbered continuously
across all groups so the user can say "fix 1 and 4":
- **Must fix:** Bugs (geometry, numerics, invariants, broken callers or
  migrations), crashes/UB/UAF, silent quality regressions, or
  over-complicated/misplaced designs that should be reworked before merging.
- **Should fix:** Long or poorly scoped functions that hurt readability, missed
  reuse of existing helpers, unnecessary code or options, duplication, scattered
  changes that can be localized, missing or bloated tests, real slowdowns,
  CI/format breaks.
- **Nit:** Small local simplifications, `const`/scoping/naming/comment cleanups.

Every finding has all four parts, each one or two short sentences:

1. **Non-reference rig sensors composed in reversed order** (`src/colmap/sfm/incremental_mapper.cc:L412-420`)
   - **What this is:** Computes `cam_from_world` for each image in a
     multi-sensor frame before triangulation.
   - **Problem:** Multiplies `frame.RigFromWorld() * rig.SensorFromRig(sensor_id)`
     instead of `rig.SensorFromRig(sensor_id) * frame.RigFromWorld()`, which
     silently passes for the identity reference sensor (`IsRefSensor`) in
     existing tests and corrupts poses for every non-reference camera.
   - **Fix:** Use `image.CamFromWorld()` (or `frame.SensorFromWorld(sensor_id)`)
     and add a non-identity `sensor_from_rig` case to
     `incremental_mapper_test.cc`.
   - **If we skip it:** Multi-camera rig triangulation uses wrong camera poses
     whenever a non-reference sensor has non-identity extrinsics.

2. **Monolithic candidate loop re-implements ray-angle filtering inline** (`src/colmap/sfm/incremental_triangulator.cc:L210-295`)
   - **What this is:** A 90-line block inside `Create` that gathers track
     observations, computes camera centers and ray angles, and filters
     candidates using mutable accumulators declared at function top.
   - **Problem:** Wide variable scopes make the control flow hard to read
     without comments, and the 25-line `acos`/dot-product block duplicates
     `CalculateTriangulationAngle` from `colmap/geometry/triangulation.h`.
   - **Fix:** Extract candidate filtering into a short anonymous-namespace
     helper (e.g., `HasSufficientTriangulationAngle`) with `const`-scoped locals
     that calls `image.ProjectionCenter()` and `CalculateTriangulationAngle`
     (~-30 lines).
   - **If we skip it:** `Create` becomes harder to read and audit, and ray-angle
     clamping diverges from `geometry/triangulation.cc`.

End with:
- `Diff reduction: ~-<N> lines possible (<1-line summary of biggest deletions/unifications>).`
  — or `Diff reduction: None obvious; diff is already tight.`
- `Verdict: Ship.` or `Verdict: Fix 1 and 2 first.`
- `Not checked:` One line, if something mattered and could not be checked (e.g.,
  HIP build, ETH3D benchmark impact).

Nothing found: `What this change does:`, `Diff reduction: None obvious; diff is already tight.`,
`Looks good. Ship.`, and one line on what you checked.
