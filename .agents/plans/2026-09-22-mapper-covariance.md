## Goal

Model covariance end-to-end through the incremental mapper: per-observation 2D measurement covariances propagate to pose and point-3D covariances, and both are consumed in whitened space by absolute pose estimation/refinement and by incremental triangulation (create, continue, merge, complete, retriangulate) and observation filtering. The pixel and angular heuristic thresholds on those paths are replaced by chi-squared gates and uncertainty-based degeneracy checks.

## Success Criteria

- Registration, triangulation, and filtering operate on whitened residuals; inlier/acceptance gates are chi-squared quantiles, not pixel/degree constants.
- Covariance loop is closed: triangulation outputs point covariances consumed by registration; bundle adjustment (plus pose refinement) outputs pose covariances consumed by triangulation.
- The replaced thresholds (`abs_pose_max_error`, `create/continue/re_max_angle_error`, `merge/complete_max_reproj_error`, `min_angle`, `filter_max_reproj_error`, `filter_min_tri_angle`) are off the default path by the end, except on the joint focal-length (P4PF) estimation path, which keeps its pixel threshold (see Phase 4).
- ETH3D DSLR A/B vs `main` (same 13-scene, 5-seed paired harness as the abs-pose-cov benchmark): no AUC/recall regression, registration recall at parity or better.
- Every new estimator has unit tests, including a synthetic known-noise calibration test where the empirical inlier rate matches the chi-squared quantile.
- Single-threaded seeded runs remain bit-deterministic.

## Context And Current Facts

- Base is `user/jsch/abs-pose-cov` @ `2622805a5`. It provides the pattern to extend: `CovariantP3PEstimator` with Mahalanobis scoring and TinySolver whitened refinement (`src/colmap/estimators/solvers/absolute_pose.h:106`), `PropagatePointCovarianceToImage` (:97), `RefineAbsolutePose` with `points3D_cov` whitening and 6x6 pose-covariance output (`src/colmap/estimators/pose.h:148`), `EstimateSchurPointCovariance` (`src/colmap/estimators/covariance.h:133`), and mapper gating via `abs_pose_use_point_covariance` (`src/colmap/sfm/incremental_mapper.h:76`, `.cc:410`).
- `BACovariance` already computes pose marginals (`GetCamCovFromWorld`, `covariance.h:38`); the mapper currently uses `Params::POINTS` only.
- `CeresBundleAdjuster` retains its problem via `Problem()` (`bundle_adjustment_ceres.h:91`), so covariances can be computed from the just-solved BA problem without a rebuild. BA default loss is `TRIVIAL` with unit weights (:16-22); an earlier attempt at hardcoded BA whitening regressed and was dropped, so BA weighting must be reintroduced only with a validated measurement model.
- RANSAC compares squared residuals against `max_error * max_error` (`src/colmap/optim/ransac.h:205`), so a chi-squared gate plugs in as `max_error = sqrt(chi2)`. Reference values computed locally: 2-DoF 95% = 5.991 (`max_error` 2.448), 99% = 9.210 (3.035), 99.9% = 13.816 (3.717).
- Triangulation threshold inventory (`IncrementalTriangulator::Options`, `incremental_triangulator.h:18-64`): `Create` uses `EstimateTriangulation` with angular error, `create_max_angle_error` 2 deg, `min_angle` 1.5 deg (`.cc:476-482`); `CompleteImage` uses reprojection error, `complete_max_reproj_error` 4 px (`.cc:150-155`); `Continue` gates angular error at `continue_max_angle_error` 2 deg (`.cc:546-548`); `Merge` gates squared reprojection at `merge_max_reproj_error` 4 px and averages by track length (`.cc:566-627`); `Complete` gates at `complete_max_reproj_error` 4 px (`.cc:664-718`); `Retriangulate` reuses `Continue` at `re_max_angle_error` 5 deg and `Create` (`.cc:286-371`).
- Filtering: `ObservationManager::FilterPoints3D` (4 px + 1.5 deg) driven by mapper `filter_max_reproj_error` / `filter_min_tri_angle` (`incremental_mapper.h:103-107`).
- 2D covariance has no current home: `Point2D` is xy-only (`scene/point2d.h`), and keypoint shapes are dropped at cache load (`database_cache.cc:231`, `FeatureKeypointsToPointsVector`). `FeatureKeypoint` carries the full affine shape plus `ComputeScale()` (`feature/types.h:37`, `feature/types.cc:110`).

## Constraints And Non-goals

- Incremental mapper only. Two-view geometry, matching, global/hierarchical mappers, and dense reconstruction are untouched (the triangulation estimator is shared code, but only the incremental call sites migrate).
- No database schema or `Point2D`/reconstruction serialization changes; covariances live in memory only.
- No pycolmap API changes; follow-up work.
- Determinism: the single-threaded seeded path must stay exactly reproducible.
- Work happens on a new branch from `user/jsch/abs-pose-cov`; no commits without explicit review (standing constraint).

## Key Decisions

1. **2D noise model: isotropic sigma from keypoint scale.** `sigma = base_sigma_px * scale^gamma` with `base_sigma_px` default 1.0 and `scale_gamma` default 0.5 (both tunable; gamma calibrated on ETH3D DSLR: sqrt mapping beat linear and fixed-sigma on the A/B grid, closing the meadow/facade gaps to near-parity with main). All plumbing carries full 2x2 covariances so an affine-shape-derived anisotropic model is a later drop-in. Rejected as default: a global constant sigma (loses the courtyard win from scale-varying weights); full-affine now (shape-convention verification and validation cost without evidence it matters first).
2. **Storage: 2D covariance on `Point2D` (single-precision), pose/point covariances in a mapper-owned cache.** Add `Eigen::Matrix2f measurement_cov` (name TBD) to `Point2D`, defaulting to identity (1 px isotropic, SPD-safe), populated from keypoint scales via the measurement model at `DatabaseCache` load. Pose and point covariances live in a new small mapper-owned class (working name `MapperCovarianceCache`) with generation stamps, since they are derived quantities that change with estimation. The new member is NOT serialized, consistent with the in-memory-only constraint; file-loaded reconstructions get the identity default. Cost is 16 bytes per `Point2D`; pycolmap bindings need no change (member simply unexposed). Rejected: a sidecar map for 2D covariances (lookup plumbing at every track/CorrData site when the payload naturally rides with the observation); serializing the member (format churn for a derived quantity).
3. **Pose covariance source: marginals from the just-solved BA problem.** After each local/global BA, run `BACovariance` in `POSES_AND_POINTS` mode on the retained `Problem()`; cache with a generation stamp. Fresh registrations seed the cache from `RefineAbsolutePose`'s 6x6 output until the next BA. This removes the "rebuilds a BA problem per registration" cost. Rejected: per-registration rebuild (too slow at triangulation scale); refinement-Hessian-only poses (ignores correlations).
4. **Triangulation estimator: `CovariantTriangulationEstimator` mirroring `CovariantP3PEstimator`.** DLT minimal seed, TinySolver whitened refinement over xyz, per-observation combined covariance `Sigma_meas + J_pose * Sigma_pose * J_pose^T` (constant-pose linearization per solve, fixed weights). RANSAC scores Mahalanobis-squared residuals against a chi-squared gate and the estimator outputs the point covariance `(J^T J)^-1` in whitened space. Rejected: weighted closed-form DLT only (no refinement or covariance output); Ceres-based refinement (heavier than the established TinySolver path).
5. **Angle-heuristic replacement: chi-squared gates plus a relative depth-uncertainty gate.** `min_angle` / create / continue angle thresholds become (a) residual chi-squared gates and (b) a scale-free degeneracy check `sigma_depth / depth <= tau` from the linearized triangulation covariance. Rejected: keeping `min_angle` alongside (defeats the goal); an absolute eigenvalue gate (scale-dependent).
6. **Merge averaging becomes information-weighted.** Replace the track-length-weighted xyz average with inverse-covariance weighting of the two point estimates; per-observation acceptance uses the whitened gate.
7. **Retriangulation looseness becomes a higher quantile.** Same whitened gates at 99.9% instead of 5 deg vs 2 deg, preserving the drift-recovery intent documented in `IncrementalMapper::Retriangulate`.
8. **BA whitening goes last.** Reintroduce `CreateCovarianceWeightedCameraCostFunction` in BA only after the measurement model is validated on the estimator paths. BA's default `TRIVIAL` loss needs no rescale in whitened units; the robust-loss path gets its scale revisited if enabled.
9. **Old thresholds stay as fallback flags during development** (one bool per phase) so any regression bisects to a phase; removed at the end.

## Recommended Approach

Build the pipeline in dependency order — measurement model and cache first (no behavior change), then the covariant triangulation estimator with synthetic calibration, then triangulator and filter migration, then the pose-covariance loop and BA weighting — so each phase is independently testable and the A/B harness gates the risky steps. Close the loop explicitly: triangulation-produced point covariances feed registration (replacing/augmenting the Schur path where fresher), and BA/refinement pose covariances feed triangulation. Keep the math first-order Gaussian throughout, matching the existing `PropagatePointCovarianceToImage` precedent, with SPD guards and NaN-safe fallbacks at every propagation site.

## Work Plan

- **Phase 0 — Branch.** Create `user/jsch/mapper-covariance` from `user/jsch/abs-pose-cov`. No code changes.
- **Phase 1 — Measurement model and `Point2D` storage (behavior-neutral).** New `estimators/measurement_noise.{h,cc}`: keypoint scale to 2x2 covariance plus chi-squared helpers. Add the single-precision 2x2 member to `Point2D` with identity default; populate at `DatabaseCache` load from keypoint scales (no serialization change). Scaffold `sfm/covariance_cache.{h,cc}` owned by `IncrementalMapper` for pose and point covariances (populated in Phase 4). Tests: SPD validity, scale monotonicity, default-member behavior, IO round-trip with defaults. Depends on: Phase 0.
- **Phase 2 — Covariant triangulation estimator.** Extend `estimators/triangulation.{h,cc}` with `CovariantTriangulationEstimator`, options (chi-squared quantile, max relative depth uncertainty), TinySolver whitened refinement, and covariance output. Keep `TriangulationEstimator` untouched. Tests: synthetic known-noise inlier-rate calibration, covariance vs Monte Carlo, numeric-Jacobian match, parallel-ray degeneracy. Depends on: Phase 1.
- **Phase 3 — Triangulator migration.** `IncrementalTriangulator::{Create, CompleteImage}` switch to the covariant estimator; `Continue`/`Merge`/`Complete` switch to whitened gates with information-weighted merge; `Retriangulate` inherits via the 99.9% quantile; `CorrData` carries covariance refs. Behind `tri_use_covariance` (default true once validated). Existing triangulator tests must pass unmodified; add gating/merge tests. Depends on: Phase 2.
- **Phase 4 — Pose-covariance loop and abs-pose measurement covariance.** After local/global BA, compute `BACovariance` (`POSES_AND_POINTS`) from the retained problem into the cache; feed pose covariances to triangulation; seed fresh registrations from `RefineAbsolutePose` output; extend `EstimateAbsolutePose`/`RefineAbsolutePose` with `points2D_cov` for combined whitening. The joint focal-length (P4PF) estimation path keeps its pixel threshold: covariance-weighted P4PF scoring needs the projection Jacobian at an unknown focal length and is a follow-up, not part of this plan. Tests: cache invalidation on BA/mutation, pose-covariance vs numeric baseline, mapper smoke test. Depends on: Phases 1-3.
- **Phase 5 — Filtering migration.** Whitened variants of `ObservationManager::FilterPoints3D*` (chi-squared per-observation gate plus relative-uncertainty gate); inject the covariance provider; `IncrementalMapper::FilterPoints` switches; old options become deprecated fallbacks. Tests: filter equivalence on easy data, outlier rejection on synthetic noise. Depends on: Phases 1, 4.
- **Phase 6 — BA whitening.** Wire measurement covariances into the Ceres BA cost functions via `CreateCovarianceWeightedCameraCostFunction`; verify `TRIVIAL`-loss neutrality and revisit robust-loss scale. Gated on ETH3D A/B parity before proceeding. Depends on: Phases 1, 5.
- **Phase 7 — Cleanup and benchmark.** Remove fallback flags and retired options; full ETH3D DSLR A/B (5+ seeds) vs `main` and vs `user/jsch/abs-pose-cov`; GUI covariance-viz spot-check. Depends on: all phases.

## Validation Plan

- Per phase: `ninja -C build` clean plus focused suites — `ctest -R "estimators/(triangulation|measurement|pose|covariance)_test"` for Phases 1-2, `ctest -R "sfm/(incremental_triangulator|incremental_mapper|observation_manager)_test"` for Phases 3-5, full `ctest` before Phase 7.
- Synthetic calibration (Phase 2, the correctness anchor): project known 3D points with sampled Gaussian pixel noise through `PerspectiveCamera`, run the covariant estimator, assert empirical inlier rate within tolerance of the chi-squared quantile and covariance agreement with Monte Carlo to ~10%.
- Mapper smoke after Phases 3-5: south-building scene must register all images with reprojection error at parity (previous baseline: 128/128, 0.51 px).
- Benchmark: reuse `/tmp/cov_ab/driver.py` + `compare.py` on ETH3D DSLR (13 scenes, 5 seeds, paired) at Phase 6 gate and Phase 7; report AUC/recall deltas with run-to-run std.
- Highest-risk validation: the Phase 6 gate — BA input weighting regressed once before, so the A/B parity check there is the go/no-go for keeping whitened BA.

## Risks / Rollback

- **Covariance-from-BA cost per local BA.** Mitigate with the retained-problem path (no rebuild) and measure mapper wall time in the benchmark; fallback is POINTS-only or sparser recomputation.
- **Miscalibrated sigma(scale).** Keep `base_sigma_px` tunable; validate against ETH3D feature-scale statistics and the synthetic calibration test before any gate depends on it.
- **Chi-squared gates too tight or loose on real data.** Quantiles stay tunable options; the A/B harness decides values, and fallback flags preserve old behavior per phase.
- **Stale pose covariances between BAs.** Generation stamps invalidate on any BA or pose mutation; missing entries fall back to large (uninformative) covariances, never to silently exact poses.
- **Rollback.** All work is branch-local; each phase is independently revertible and bisectable via its fallback flag.

## Open Questions

- **O1 — Post-solve problem validity for `EstimateBACovariance`.** `Problem()` access exists, but confirm the retained problem is valid input after `Solve()` in Phase 4 (expected yes; fallback is an explicit problem rebuild for covariance only).
- **O2 — Sigma(scale) calibration constant.** Set empirically in Phase 1 from feature-scale statistics and synthetic tests; no principled value exists yet.
- None other; all other material facts were verified in the workspace during planning.
