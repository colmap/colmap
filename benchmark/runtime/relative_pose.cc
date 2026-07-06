// Benchmark comparing local-optimization strategies for calibrated two-view
// relative pose (essential matrix) estimation, on synthetic data:
//
//   * COLMAP  : EstimateRelativePose(), i.e. LO-RANSAC with the current
//               *linear* local re-fit (5-point solver re-run on the inlier set,
//               no non-linear / Sampson refinement).
//   * PoseLib : estimate_relative_pose(), i.e. LO-RANSAC with PoseLib's
//               bespoke, fixed-size Levenberg-Marquardt Sampson refinement.
//
// Both estimators are driven with matched inlier thresholds and RANSAC budgets
// so that the main difference under test is the local-optimization step. For
// each configuration we report wall-clock time (Google Benchmark) plus one-shot
// accuracy counters (median/mean rotation and translation-direction error, and
// failure rate) computed over a batch of synthetic problems.
//
// NOTE on threshold parity: COLMAP computes the Sampson error on unit bearing
// rays, whereas PoseLib computes it on homogeneous normalized image points
// (u, v, 1). For a modest field of view (|u|,|v| < ~0.7) the two differ by at
// most ~1.5x, so inlier sets are close but not identical. We keep the FOV
// modest to minimize this discrepancy; treat absolute inlier counts as
// approximate, and the rotation/translation errors (vs. ground truth) as the
// primary accuracy signal.

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/util/eigen_alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>
#include <PoseLib/types.h>
#include <benchmark/benchmark.h>

using namespace colmap;

// Defined in the (locally instrumented) PoseLib build; counts cheirality
// checks, model-scoring calls, and total points scored during MSAC scoring.
namespace poselib {
extern unsigned long long g_poselib_cheirality_calls;
extern unsigned long long g_poselib_score_calls;
extern unsigned long long g_poselib_point_evals;
}  // namespace poselib
// Defined in colmap_geometry; counts ray-based ComputeSquaredSampsonError calls
// (model-scorings) and total points scored.
namespace colmap {
extern unsigned long long g_colmap_score_calls;
extern unsigned long long g_colmap_point_evals;
}  // namespace colmap

namespace {

// ---------------------------------------------------------------------------
// Synthetic data generation.
// ---------------------------------------------------------------------------

// Common inlier threshold (Sampson error in normalized image coordinates) and
// per-point noise, shared by both estimators for a fair comparison.
constexpr double kMaxError = 0.005;     // ~5px at focal length 1000.
constexpr double kInlierNoise = 0.002;  // ~2px at focal length 1000.
constexpr int kNumProblems = 100;       // Batch size for accuracy statistics.
constexpr unsigned kSeed = 42;

struct Problem {
  Rigid3d gt_cam2_from_cam1;
  // Unit bearing rays (COLMAP input).
  std::vector<Eigen::Vector3d> rays1;
  std::vector<Eigen::Vector3d> rays2;
  // Normalized image points p = ray.hnormalized() (PoseLib input).
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
};

Eigen::Quaterniond RandomSmallRotation() {
  // Uniform-ish random rotation via random axis-angle with bounded angle so
  // both views look roughly forward.
  const Eigen::Vector3d axis(RandomUniformReal<double>(-1.0, 1.0),
                             RandomUniformReal<double>(-1.0, 1.0),
                             RandomUniformReal<double>(-1.0, 1.0));
  const double angle = RandomUniformReal<double>(0.0, 0.5);  // radians.
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis.normalized()));
}

Problem GenerateProblem(int num_points, double inlier_ratio) {
  Problem problem;
  const Eigen::Vector3d gt_translation(RandomUniformReal<double>(-1.0, 1.0),
                                       RandomUniformReal<double>(-1.0, 1.0),
                                       RandomUniformReal<double>(-1.0, 1.0));
  problem.gt_cam2_from_cam1 =
      Rigid3d(RandomSmallRotation(), gt_translation.normalized());

  problem.rays1.reserve(num_points);
  problem.rays2.reserve(num_points);
  problem.points1.reserve(num_points);
  problem.points2.reserve(num_points);

  for (int i = 0; i < num_points; ++i) {
    // Sample a 3D point that is in front of BOTH cameras. It is always in front
    // of camera 1 by construction (positive depth along a forward-facing ray),
    // but the small-rotation + unit-baseline setup does not guarantee positive
    // depth in camera 2. Reject draws that fall behind camera 2 so that inlier
    // correspondences stay cheirality-consistent instead of being silently
    // sign-flipped into forward-facing rays by hnormalized().
    Eigen::Vector2d p1;
    Eigen::Vector3d point3D_in_cam2;
    do {
      // Normalized image point in the first camera, modest FOV.
      p1 = Eigen::Vector2d(RandomUniformReal<double>(-0.7, 0.7),
                           RandomUniformReal<double>(-0.7, 0.7));
      const Eigen::Vector3d ray1 = p1.homogeneous().normalized();
      const double depth = RandomUniformReal<double>(2.0, 5.0);
      point3D_in_cam2 = problem.gt_cam2_from_cam1 * (depth * ray1);
    } while (point3D_in_cam2.z() <= 0.0);
    Eigen::Vector2d p2 = point3D_in_cam2.hnormalized();

    // Add Gaussian pixel noise (in normalized coordinates) to both views.
    Eigen::Vector2d p1_noisy = p1;
    p1_noisy.x() += RandomGaussian<double>(0.0, kInlierNoise);
    p1_noisy.y() += RandomGaussian<double>(0.0, kInlierNoise);

    if (RandomUniformReal<double>(0.0, 1.0) > inlier_ratio) {
      // Gross outlier: random correspondence in the second view.
      p2 = Eigen::Vector2d(RandomUniformReal<double>(-0.7, 0.7),
                           RandomUniformReal<double>(-0.7, 0.7));
    } else {
      p2.x() += RandomGaussian<double>(0.0, kInlierNoise);
      p2.y() += RandomGaussian<double>(0.0, kInlierNoise);
    }

    problem.points1.push_back(p1_noisy);
    problem.points2.push_back(p2);
    problem.rays1.push_back(p1_noisy.homogeneous().normalized());
    problem.rays2.push_back(p2.homogeneous().normalized());
  }
  return problem;
}

std::vector<Problem> GenerateProblems(int num_points, double inlier_ratio) {
  SetPRNGSeed(kSeed);
  std::vector<Problem> problems;
  problems.reserve(kNumProblems);
  for (int i = 0; i < kNumProblems; ++i) {
    problems.push_back(GenerateProblem(num_points, inlier_ratio));
  }
  return problems;
}

// ---------------------------------------------------------------------------
// Error metrics.
// ---------------------------------------------------------------------------

double RotationErrorDeg(const Rigid3d& gt, const Rigid3d& est) {
  const Eigen::Quaterniond dq = gt.rotation() * est.rotation().inverse();
  const double angle = 2.0 * std::atan2(dq.vec().norm(), std::abs(dq.w()));
  return angle * 180.0 / M_PI;
}

// Translation is only defined up to scale (and the sign is convention
// dependent), so we report the angle between the (normalized) directions.
double TranslationErrorDeg(const Rigid3d& gt, const Rigid3d& est) {
  const Eigen::Vector3d t_gt = gt.translation().normalized();
  const Eigen::Vector3d t_est = est.translation().normalized();
  const double c = std::min(1.0, std::abs(t_gt.dot(t_est)));
  return std::acos(c) * 180.0 / M_PI;
}

double Median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  return v[mid];
}

double Mean(const std::vector<double>& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  double s = 0.0;
  for (double x : v) s += x;
  return s / v.size();
}

// ---------------------------------------------------------------------------
// Estimator wrappers.
// ---------------------------------------------------------------------------

// Variant of EssentialMatrixLMEstimator that differs in exactly one respect: it
// scores support with the plain (algebraic) Sampson error instead of the
// cheirality-aware residual used by the base estimator. Everything else - the
// minimal-sample count and the self-seeding Estimate() / initial-model Refine()
// that LO-RANSAC drives - is inherited unchanged, so comparing this against the
// base estimator isolates the effect of the cheirality check.
struct EssentialMatrixLMEstimatorNoCheirality
    : public EssentialMatrixLMEstimator {
  static void Residuals(const std::vector<X_t>& cam_rays1,
                        const std::vector<Y_t>& cam_rays2,
                        const M_t& E,
                        std::vector<double>* residuals) {
    ComputeSquaredSampsonError(cam_rays1, cam_rays2, E, residuals);
  }
};

// Variant of the five-point minimal estimator that gates its hypotheses by
// cheirality on the sample rays: an essential matrix is only returned if the
// five sample correspondences triangulate in front of both cameras. This
// mirrors how PoseLib's relpose_5pt (CameraPose overload) filters via
// motion_from_essential, and it prunes geometrically impossible hypotheses -
// dominant on contaminated samples - before they reach the expensive full-N
// scoring, cutting scoring *volume* rather than per-scoring cost.
struct EssentialMatrixFivePointGatedEstimator
    : public EssentialMatrixFivePointEstimator {
  static void Estimate(const std::vector<X_t>& cam_rays1,
                       const std::vector<Y_t>& cam_rays2,
                       std::vector<M_t>* models) {
    std::vector<M_t> candidates;
    EssentialMatrixFivePointEstimator::Estimate(
        cam_rays1, cam_rays2, &candidates);
    models->clear();
    Rigid3d cam2_from_cam1;
    std::vector<int> valid_indices;
    for (const M_t& E : candidates) {
      PoseFromEssentialMatrix(
          E, cam_rays1, cam_rays2, &cam2_from_cam1, &valid_indices);
      // Keep the hypothesis only if every sample correspondence is in front of
      // both cameras for the recovered pose.
      if (valid_indices.size() == cam_rays1.size()) {
        models->push_back(E);
      }
    }
  }
};

// RANSAC-loop statistics surfaced for diagnostics (NaN if not applicable).
struct RunStats {
  double num_trials = std::numeric_limits<double>::quiet_NaN();
  double num_inliers = std::numeric_limits<double>::quiet_NaN();
  // Scoring-work counters (set per engine below).
  double num_score_calls = std::numeric_limits<double>::quiet_NaN();
  double num_point_evals = std::numeric_limits<double>::quiet_NaN();
  // PoseLib-only diagnostics (NaN for COLMAP methods).
  double num_refinements = std::numeric_limits<double>::quiet_NaN();
  double num_cheirality = std::numeric_limits<double>::quiet_NaN();
};

// COLMAP LO-RANSAC with configurable minimal + local (LO) estimators and
// support measure.
template <typename MinimalEstimator,
          typename LocalEstimator,
          typename SupportMeasurer = InlierSupportMeasurer>
bool RunColmapRansac(const Problem& problem, Rigid3d* est, RunStats* stats) {
  RANSACOptions options;
  options.max_error = kMaxError;
  options.confidence = 0.9999;
  options.min_num_trials = 1000;
  options.max_num_trials = 100000;
  options.random_seed = 0;

  LORANSAC<MinimalEstimator, LocalEstimator, SupportMeasurer> ransac(options);
  const unsigned long long score_calls_before = g_colmap_score_calls;
  const unsigned long long point_evals_before = g_colmap_point_evals;
  const auto report = ransac.Estimate(problem.rays1, problem.rays2);
  if (stats != nullptr) {
    stats->num_trials = static_cast<double>(report.num_trials);
    stats->num_inliers = static_cast<double>(report.support.num_inliers);
    stats->num_score_calls =
        static_cast<double>(g_colmap_score_calls - score_calls_before);
    stats->num_point_evals =
        static_cast<double>(g_colmap_point_evals - point_evals_before);
  }
  if (!report.success) {
    return false;
  }

  std::vector<Eigen::Vector3d> inlier_rays1;
  std::vector<Eigen::Vector3d> inlier_rays2;
  inlier_rays1.reserve(report.support.num_inliers);
  inlier_rays2.reserve(report.support.num_inliers);
  for (size_t i = 0; i < problem.rays1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inlier_rays1.push_back(problem.rays1[i]);
      inlier_rays2.push_back(problem.rays2[i]);
    }
  }
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(
      report.model, inlier_rays1, inlier_rays2, est, &valid_indices);
  if (est->rotation().coeffs().array().isNaN().any() ||
      est->translation().array().isNaN().any()) {
    return false;
  }
  if (valid_indices.empty()) {
    return false;
  }
  return true;
}

bool RunPoseLib(const Problem& problem,
                const poselib::Camera& camera,
                Rigid3d* est,
                RunStats* stats = nullptr) {
  poselib::RelativePoseOptions options;
  options.max_error = kMaxError;
  options.ransac.success_prob = 0.9999;
  options.ransac.min_iterations = 1000;
  options.ransac.max_iterations = 100000;
  options.ransac.seed = 0;
  poselib::CameraPose pose;
  std::vector<char> inliers;
  const unsigned long long cheirality_before =
      poselib::g_poselib_cheirality_calls;
  const unsigned long long score_calls_before = poselib::g_poselib_score_calls;
  const unsigned long long point_evals_before = poselib::g_poselib_point_evals;
  const poselib::RansacStats pl_stats =
      poselib::estimate_relative_pose(problem.points1,
                                      problem.points2,
                                      camera,
                                      camera,
                                      options,
                                      &pose,
                                      &inliers);
  if (stats != nullptr) {
    stats->num_trials = static_cast<double>(pl_stats.iterations);
    stats->num_inliers = static_cast<double>(pl_stats.num_inliers);
    stats->num_refinements = static_cast<double>(pl_stats.refinements);
    stats->num_cheirality = static_cast<double>(
        poselib::g_poselib_cheirality_calls - cheirality_before);
    stats->num_score_calls = static_cast<double>(
        poselib::g_poselib_score_calls - score_calls_before);
    stats->num_point_evals = static_cast<double>(
        poselib::g_poselib_point_evals - point_evals_before);
  }
  if (pl_stats.num_inliers == 0) {
    return false;
  }
  *est = ConvertPoseLibPoseToRigid3d(pose);
  return true;
}

// ---------------------------------------------------------------------------
// Benchmarks.
// ---------------------------------------------------------------------------

// The colmap methods form a 2x2 over {5-point algebraic LO, in-loop LM LO} x
// {InlierSupportMeasurer (count-primary), MEstimatorSupportMeasurer (MSAC)}.
//
// kColmap     : 5-point minimal + 5-point algebraic LO, count support
// (shipping). kColmap8pt  : 5-point minimal + 8-point linear LO, count support.
// kColmapMsac : 5-point minimal + 5-point algebraic LO, MSAC support.
// kColmapLM   : 5-point minimal + in-loop EssentialMatrixLMEstimator LO, count
//               support (LM refit handicapped by the count-primary measure).
// kColmapLMMsac: 5-point minimal + in-loop EssentialMatrixLMEstimator LO, MSAC
//               support (the intended configuration, cheirality-aware
//               residual).
// kColmapLMMsacNoCheiral: identical to kColmapLMMsac but the LO estimator
// scores
//               support with the plain Sampson error (no cheirality), isolating
//               the effect of the cheirality-aware residual.
// kColmapGated: identical to kColmap but the minimal + local estimators gate
//               five-point hypotheses by sample-ray cheirality (PoseLib-style),
//               pruning geometrically impossible models before scoring.
// kPoseLib    : PoseLib robust estimator with its LM Sampson refinement.
enum class Method {
  kColmap,
  kColmap8pt,
  kColmapMsac,
  kColmapLM,
  kColmapLMMsac,
  kColmapLMMsacNoCheiral,
  kColmapGated,
  kPoseLib
};

bool Run(Method method,
         const Problem& problem,
         const poselib::Camera& camera,
         Rigid3d* est,
         RunStats* stats = nullptr) {
  switch (method) {
    case Method::kColmap:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixFivePointEstimator>(
          problem, est, stats);
    case Method::kColmap8pt:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixEightPointEstimator>(
          problem, est, stats);
    case Method::kColmapMsac:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixFivePointEstimator,
                             MEstimatorSupportMeasurer>(problem, est, stats);
    case Method::kColmapLM:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixLMEstimator>(problem, est, stats);
    case Method::kColmapLMMsac:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixLMEstimator,
                             MEstimatorSupportMeasurer>(problem, est, stats);
    case Method::kColmapLMMsacNoCheiral:
      return RunColmapRansac<EssentialMatrixFivePointEstimator,
                             EssentialMatrixLMEstimatorNoCheirality,
                             MEstimatorSupportMeasurer>(problem, est, stats);
    case Method::kColmapGated:
      // Gate only the minimal estimator; the local (LO) estimator refits on the
      // full inlier set, where an "all points in front" gate would spuriously
      // reject valid refits and disable local optimization.
      return RunColmapRansac<EssentialMatrixFivePointGatedEstimator,
                             EssentialMatrixFivePointEstimator>(
          problem, est, stats);
    case Method::kPoseLib:
      return RunPoseLib(problem, camera, est, stats);
  }
  return false;
}

void ReportAccuracy(benchmark::State& state,
                    const std::vector<Problem>& problems,
                    Method method,
                    const poselib::Camera& camera) {
  std::vector<double> rot_errs;
  std::vector<double> trans_errs;
  std::vector<double> num_trials;
  std::vector<double> inlier_pcts;
  std::vector<double> refinements;
  std::vector<double> cheirality;
  std::vector<double> score_calls;
  std::vector<double> point_evals;
  int num_failures = 0;
  for (const Problem& problem : problems) {
    Rigid3d est;
    RunStats run_stats;
    const bool ok = Run(method, problem, camera, &est, &run_stats);
    if (!std::isnan(run_stats.num_trials)) {
      num_trials.push_back(run_stats.num_trials);
    }
    if (!std::isnan(run_stats.num_refinements)) {
      refinements.push_back(run_stats.num_refinements);
    }
    if (!std::isnan(run_stats.num_cheirality)) {
      cheirality.push_back(run_stats.num_cheirality);
    }
    if (!std::isnan(run_stats.num_score_calls)) {
      score_calls.push_back(run_stats.num_score_calls);
    }
    if (!std::isnan(run_stats.num_point_evals)) {
      point_evals.push_back(run_stats.num_point_evals);
    }
    if (!ok) {
      ++num_failures;
      continue;
    }
    rot_errs.push_back(RotationErrorDeg(problem.gt_cam2_from_cam1, est));
    trans_errs.push_back(TranslationErrorDeg(problem.gt_cam2_from_cam1, est));
    // Recovered inliers as a percentage of all correspondences.
    if (!std::isnan(run_stats.num_inliers)) {
      inlier_pcts.push_back(100.0 * run_stats.num_inliers /
                            static_cast<double>(problem.rays1.size()));
    }
  }
  state.counters["rot_err_med_deg"] = Median(rot_errs);
  state.counters["rot_err_mean_deg"] = Mean(rot_errs);
  state.counters["t_err_med_deg"] = Median(trans_errs);
  state.counters["t_err_mean_deg"] = Mean(trans_errs);
  state.counters["num_trials_mean"] = Mean(num_trials);
  state.counters["inlier_pct_mean"] = Mean(inlier_pcts);
  state.counters["refinements_mean"] = Mean(refinements);
  state.counters["cheirality_mean"] = Mean(cheirality);
  state.counters["score_calls_mean"] = Mean(score_calls);
  state.counters["point_evals_mean"] = Mean(point_evals);
  state.counters["fail_pct"] =
      100.0 * num_failures / static_cast<double>(problems.size());
}

void BM_RelativePose(benchmark::State& state,
                     Method method,
                     int num_points,
                     double inlier_ratio) {
  const std::vector<Problem> problems =
      GenerateProblems(num_points, inlier_ratio);
  const poselib::Camera camera("SIMPLE_PINHOLE", {1.0, 0.0, 0.0}, 0, 0);

  for (auto _ : state) {
    for (const Problem& problem : problems) {
      Rigid3d est;
      Run(method, problem, camera, &est);
      benchmark::DoNotOptimize(est);
    }
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(problems.size()));

  // One-shot accuracy pass (untimed).
  ReportAccuracy(state, problems, method, camera);
}

// Registers all methods back-to-back for each {num_points, inlier_ratio} case,
// so the output interleaves the methods per case rather than grouping by
// method.
int RegisterBenchmarks() {
  for (int num_points : {100, 500, 1000}) {
    for (int inlier_pct : {100, 80, 50}) {
      const double inlier_ratio = inlier_pct / 100.0;
      const std::string suffix =
          "/" + std::to_string(num_points) + "/" + std::to_string(inlier_pct);
      benchmark::RegisterBenchmark("colmap" + suffix,
                                   BM_RelativePose,
                                   Method::kColmap,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap8pt" + suffix,
                                   BM_RelativePose,
                                   Method::kColmap8pt,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap_msac" + suffix,
                                   BM_RelativePose,
                                   Method::kColmapMsac,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap_lm" + suffix,
                                   BM_RelativePose,
                                   Method::kColmapLM,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap_lm_msac" + suffix,
                                   BM_RelativePose,
                                   Method::kColmapLMMsac,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap_lm_msac_nocheiral" + suffix,
                                   BM_RelativePose,
                                   Method::kColmapLMMsacNoCheiral,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("colmap_gated" + suffix,
                                   BM_RelativePose,
                                   Method::kColmapGated,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark("poselib" + suffix,
                                   BM_RelativePose,
                                   Method::kPoseLib,
                                   num_points,
                                   inlier_ratio)
          ->Unit(benchmark::kMillisecond);
    }
  }
  return 0;
}

[[maybe_unused]] const int kRegistered = RegisterBenchmarks();

}  // namespace

BENCHMARK_MAIN();
