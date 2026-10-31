---
name: colmap-review
description: >-
  Maintainer-grade code review for COLMAP (C++, CUDA/HIP, pycolmap, CMake, and
  benchmarks): checks geometric and numerical correctness, memory/binding
  safety, performance, and test quality, while aggressively finding ways to
  shrink the diff, eliminate duplication, and localize changes to fewer places.
  Use for "review this", "code review", "review the last commit", "review my
  PR", "can this diff be smaller", "is this over-engineered", or /colmap-review.
  Don't use when asked to implement changes or run benchmarks directly.
---

Review a COLMAP change like a core maintainer who will own every added line for
years. Order of importance: **correct**, **lean & localized**, **safe**,
**fast**, **tested**, **consistent**. Every unnecessary line, option, file, or
duplicate branch is a permanent maintenance burden — actively hunt for ways to
reduce total diff size and confine changes to as few code locations as possible.
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
    default, enum, database schema, or deprecation changes. Broken or stale call
    sites often sit outside the diff.
  - Read sibling implementations in the touched module (other camera models,
    solvers, cost functors, controllers, `pycolmap` bindings, `*_test.cc` files)
    to spot existing patterns and helpers the diff should reuse.
  - Trace data and coordinate frames end-to-end: where poses, intrinsics,
    observations, and IDs enter, what invariants hold, and what mutates in
    place.

## 2. Look for

1. **Bloat, scattered changes & duplication (Shrink the diff):**
   - **Delete / question existence:** Dead code, unused helpers or `#include`s,
     debug `LOG(INFO)` noise, speculative options/callbacks nobody sets, fixes
     to unused scripts, analytical Jacobians or special-case fast paths where
     Ceres autodiff or the general path suffices without a measured win, and
     unrelated changes bundled into the PR.
   - **Localize to fewer locations:** Scattered `if/else` plumbing across
     callers that belongs behind one shared abstraction or existing hook — e.g.,
     integrating with `Rig`/`sensor_t`/`data_t` instead of bolting on parallel
     IDs, adding a flag/field on `PoseGraph::Edge` or `TwoViewGeometry` instead
     of threading parallel structures, exposing a `Camera` model trait instead
     of ad-hoc parameter-count checks in callers, extracting one private
     `Clear()`/reset helper instead of updating both `ReadText` and
     `ReadBinary`, or using a `-1` sentinel default resolved in one place
     instead of syncing `std::optional` defaults across `OptionManager` and
     estimator structs.
   - **Reuse COLMAP building blocks:** Existing cost wrappers
     (`CovarianceWeightedCostFunctor`, `ScaleWeightedCostFunctor`, generic prior
     functors), `MAKE_ENUM`/`MAKE_ENUM_CLASS`, `OptionManager`,
     `ScopedSignalHandler`/`PyInterruptChecker`,
     `AddCameraWithTrivialRig`/`AddImageWithTrivialFrame`, and compact Eigen
     idioms (`.homogeneous().normalized()`, `.cast<T>().homogeneous()`,
     `.diagonal()`, `.asDiagonal()`, `Eigen::Matrix2x3d`, `.value_or()`). Merge
     near-copies (duplicate `Estimate*` dispatch, minimal/non-minimal paths).
   - **Keep designs flat & headers small:** Prefer a free function over a
     single-method class, a `struct` over a trivial `class`, static
     `CreateDefault()` factories over `friend` functions, return values (`T` or
     `std::optional<T>`) over out-parameters, and flat `*Options` structs (no
     inheritance). Move non-trivial bodies, `operator<<`, and internal
     constants/helpers out of `.h` into `.cc` (anonymous namespace).

2. **Geometric, numerical & SfM/MVS bugs:**
   - **Camera model generality:** Assuming a finite perspective plane
     (`CamFromImg`, `z = 1`, non-empty `FocalLengthIdxs()`) when fisheye rays
     (`>= 90°`) or `SPHERICAL` back-hemisphere rays are valid — use unit
     bearings (`CamRayFromImg` / `CamRayFromImgWithJac`). Check `±180°` seam
     wrap-around on spherical cameras, and continuous pixel-center (`+0.5`) vs.
     integer-grid (`Bitmap::InterpolateBilinear`) `0.5`px offsets.
   - **Residuals, solvers & degeneracies:** Refinement cost functors assigning
     `0` residual to failed/behind-camera projections (scoring invalid poses as
     perfect fits) or returning `false` on a single outlier observation
     (aborting the Ceres step instead of penalizing); `.normalized()` on zero
     fallback rays producing `0/0 = NaN`; `LLT`/`LDLT` without
     positive-definiteness checks (`LDLT::info() == Success` does not guarantee
     PD); scale-dependent determinant thresholds instead of scale-free
     singular-value ratios; unobservable scale on central/panoramic rigs (check
     the *final RANSAC / LO-refinement inliers*, not only all inputs);
     `TinySolver` writebacks ignoring `NUMERICAL_FAILURE` or `!allFinite()`;
     gimbal-lock / small-angle singularities; `double` precision loss on
     `timestamp_t` (`> 2^53` ns).
   - **Frames, manifolds & determinism:** Missing Ceres manifolds
     (`EigenQuaternionManifold`, `SphereManifold`, positive/log scale — missing
     manifolds also overrun covariance buffers); wrong `target_from_source`
     (`cam_from_world`, `rig_from_world`, `sensor_from_rig`) composition, left-
     vs. right-perturbation Jacobians, or `Invert` conjugation; rig calibration
     state (`NaN` translation = unknown, ref vs. non-ref sensor, backend limits
     like Caspar ignoring `constant_rig_from_world_rotation`); iterating
     `FlatHashMap`/`FlatHashSet` when picking spanning-tree hubs, gauge-fixed
     frames, or output order (nondeterministic across runs); accidental PRNG
     reseeding.

3. **Safety, error contracts & cross-language (`pycolmap` / CMake) risks:**
   - **State & database safety:** Validate preconditions *before* mutating
     `Reconstruction` or `Database` so exceptions never leave partial state. Use
     `THROW_CHECK*` / `.at()` for violated contracts and unchecked indexing,
     exceptions (not `LOG(FATAL)`) in factories, and `LOG(WARNING)` +
     skip/`HasBogusParams` filtering rather than crashing a pipeline on one
     diverged camera. In SQLite changes, check `ALTER TABLE RENAME` index loss,
     orphaned rows without FKs, `MAX+1` ID reuse, fragile bare-column `MAX()`
     queries, and `static_assert`s on persisted enum integers.
   - **`pycolmap` bindings:** Raw pointers or NumPy views into mutable
     containers (`bind_map` `__delitem__`/`clear()` causing use-after-free in
     Ceres/Python); raw `T*` arguments without `None` handling (segfault); C++
     mutating `std::vector<T>*` across pybind11's copy-converting STL boundary
     (updates silently lost in Python); `std::function` fields breaking options
     pickling; missing `__copy__`/`__deepcopy__` alongside copy constructors;
     inconsistent GIL release; wrong bound method; missing option parity or
     docstrings across C++, `OptionManager` CLI, GUI, and `pycolmap`.
   - **Build & CI portability:** `#if defined(COLMAP_CUDA_ENABLED)` missing
     `COLMAP_HIP_ENABLED` / `COLMAP_GPU_ENABLED` or headless (`GUI_ENABLED=OFF`)
     paths; simultaneous CUDA+HIP builds; GPU tests missing runtime
     `GTEST_SKIP()` when `GetNumCudaDevices() == 0`; missing exports in
     `cmake/colmap-config.cmake.in`; hardcoded `lib` vs. `CMAKE_INSTALL_LIBDIR`;
     Windows path separators; formatting/lint failures (`scripts/format/c++.sh`,
     `scripts/format/python.sh`, `.clang-tidy` `-Werror`, GCC
     `-Werror=range-loop-construct`).

4. **Performance & algorithmic evidence:**
   - Flag `O(N^2)` behavior (single-item sorted `Insert` in a loop instead of
     batch sort/insert, `Reconstruction` hash lookups inside `std::sort`
     comparators), redundant image I/O or extra SQLite queries, and avoidable
     hot-loop work (copying Eigen blocks/loop pairs instead of `const auto&`,
     not taking sink vectors by value + `std::move` or reusing output buffers,
     `.norm()` instead of `.squaredNorm()`, `std::pow(x, 2)` instead of `x * x`,
     `x / 2.0` instead of `0.5 * x`, recomputing loop-invariant
     transposes/trig).
   - Algorithmic or default-parameter changes in SfM, MVS, or RANSAC require
     empirical justification (accuracy + runtime on benchmarks such as ETH3D or
     multi-seed synthetic sweeps). Push back on unverified default changes or
     inner-loop slowdowns.

5. **Missing tests & test bloat:**
   - **Missing or vacuous tests:** Risky logic, new solvers/cost functors,
     `operator<<`, or bug fixes need a focused C++ test (`*_test.cc`) that fails
     without the change. Catch vacuous assertions: `len(models) == len(ref)`
     without checking `num_reg_frames() > 0` / `num_points3D() > 0`, bare
     `EXPECT_ANY_THROW` instead of `testing::HasSubstr(...)`, cancellation/skip
     flags with no baseline proving they took effect, or round-trip checks that
     miss documented conventions at singularities.
   - **Shrink test code:** Functional tests belong in C++ (`*_test.cc`) — keep
     `pycolmap` tests (`*_test.py`) strictly focused on the binding layer
     (`None`/empty inputs, types, lifetime) unless non-trivial logic lives in
     Python. Delete redundant tests that add no branch coverage; merge
     repetitive cases into `TEST_P` or table-driven loops; use
     `ReconstructionEq`, `Rigid3dNear`, `Sim3dNear`, `EigenMatrixNear`,
     `SynthesizeDataset`, and `Database::Open(kInMemorySqliteDatabasePath)`
     instead of verbose field loops, temp directories, or slow end-to-end
     pipeline runs; gate heavy inference tests behind `COLMAP_TEST_HEAVY`.

6. **COLMAP conventions (flag concrete deviations, never personal taste):**
   - `FlatHashMap`/`FlatHashSet` (default; no iterator-erase loops) and
     `NodeHashMap`/`NodeHashSet` (when pointer/ref stability across mutations is
     needed), never `std::unordered_map/set`; `std::map/set` only when sorted
     order is required (determinism, Ceres parameter block order).
   - Strong IDs (`camera_t`, `image_t`, `frame_t`, `rig_t`, `sensor_t`,
     `point2D_t`, `point3D_t`, etc.); `target_from_source` and `x_in_y` naming
     (`cam_ray`, `cam_point`, `img_point`; camera `Extra` params, not
     `Distortion`); output pointer/ref params last; strict `const` on locals and
     immutable members; `constexpr kPascalCase`; `EIGEN_PI` and
     `std::numeric_limits` (not `M_PI`/`DBL_MAX`); `//` comments with
     `@param`/`@return` API docs (not `///`);
     `// SPDX-License-Identifier: BSD-3-Clause` on new files.

## 3. Check before you report

- **Concrete case or concrete deletion:** Every bug, safety, or performance
  finding needs a concrete trigger and failure ("when `CamRayFromImg` fails,
  `Zero().normalized()` produces `NaN` in `A`"). Every lean/maintainability
  finding needs a verified simplification ("replace `WeightedBATA...` with
  `CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>` in
  `motion_averaging.h`, saving ~45 lines"). No case, no finding.
- **Verify against the repo:** Re-read the lines, callers, and any COLMAP helper
  or matcher you suggest reusing to confirm exact signatures, visibility, and
  module-layer rules. When a `build/` directory or Python environment is
  available and a subtle numerical, SQLite, or formatting question arises,
  verify with a quick read-only check (targeted `ctest` or scratch calculation)
  — never modify or leave untracked files in the workspace.
- **Smallest fix wins:** Prefer fixes that delete code, reuse existing COLMAP
  primitives, and touch fewer files. Never propose new layers, base classes, or
  options unless collapsing real duplication.
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
appropriately localized.

Then list findings in three groups (skip empty groups), numbered continuously
across all groups so the user can say "fix 1 and 4":
- **Must fix:** Bugs (geometry, numerics, invariants, broken callers or
  migrations), crashes/UB/UAF, silent quality regressions, or
  over-complicated/misplaced designs that should be reworked before merging.
- **Should fix:** Unnecessary code or options, duplication, scattered changes
  that can be localized, missing or bloated tests, real slowdowns, undocumented
  preconditions, CI/format breaks.
- **Nit:** Small local simplifications, `const`/naming/idiom cleanups.

Every finding has all four parts, each one or two short sentences:

1. **Points behind the camera score as zero error in refinement** (`src/colmap/estimators/cost_functions/reprojection_error.h:L447-457`)
   - **What this is:** Ceres cost functor for refining scaled generalized
     absolute pose over rig observations.
   - **Problem:** When `ImgFromCam` fails (point behind camera), the functor
     writes a `0` residual instead of penalizing or rejecting, so a pose with
     all points behind the rig is accepted as a perfect fit.
   - **Fix:** Use a bearing/angular residual or finite cheirality penalty
     consistent with RANSAC scoring, and add a behind-camera regression case in
     `generalized_absolute_pose_test.cc`.
   - **If we skip it:** Nonlinear refinement can converge to or preserve
     inverted poses with zero reported cost.

2. **Duplicate model-reset logic and oversized pycolmap test** (`src/colmap/scene/reconstruction.cc:L812-845`, `src/pycolmap/scene/reconstruction_test.py:L210-275`)
   - **What this is:** Resets reconstruction bookkeeping in `ReadText` and
     `ReadBinary`, plus a 65-line Python reload pipeline test.
   - **Problem:** The 8-field reset block is copied in two loaders, and the
     functional test duplicates C++ coverage in `pycolmap` using temporary files
     on disk.
   - **Fix:** Extract a private `Reconstruction::Clear()` called by both
     readers, assert reload state in `reconstruction_test.cc` via
     `ReconstructionEq`, and drop the Python test (~-70 lines, 2 files instead
     of 3).
   - **If we skip it:** Any new `Reconstruction` member must be remembered in
     multiple loaders, and `pycolmap` test runtime and maintenance grow without
     extra binding coverage.

End with:
- `Diff reduction: ~-<N> lines possible (<1-line summary of biggest deletions/unifications>).`
  — or `Diff reduction: None obvious; diff is already tight.`
- `Verdict: Ship.` or `Verdict: Fix 1 and 2 first.`
- `Not checked:` One line, if something mattered and could not be checked (e.g.,
  HIP build, ETH3D benchmark impact).

Nothing found: `What this change does:`, `Diff reduction: None obvious; diff is already tight.`,
`Looks good. Ship.`, and one line on what you checked.
