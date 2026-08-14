// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// A/B comparison of plain LO-RANSAC vs DEGENSAC for fundamental matrix
// estimation on synthetic two-view data with a dominant scene plane. Prints a
// markdown table of accuracy, robustness, and runtime metrics across scene
// configurations, in the style of the essential-matrix cheirality benchmark.
//
// For each configuration a number of independent problems are generated. Both
// methods see the identical data and identical RANSAC randomness per problem,
// so the comparison is apples-to-apples. Pose error is measured by decomposing
// the estimated fundamental matrix into a relative pose and comparing it to
// ground truth.

#include "colmap/estimators/fundamental_matrix_degensac.h"

#include "colmap/estimators/solvers/fundamental_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace colmap;

namespace {

constexpr double kMaxError = 1.0;  // Inlier pixel threshold.

Eigen::Matrix3d RandomCalibrationMatrix() {
  return (Eigen::Matrix3d() << RandomUniformReal<double>(800, 1200),
          0,
          RandomUniformReal<double>(400, 600),
          0,
          RandomUniformReal<double>(800, 1200),
          RandomUniformReal<double>(400, 600),
          0,
          0,
          1)
      .finished();
}

struct Scene {
  Eigen::Matrix3d K;
  Rigid3d cam2_from_cam1;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  // True inliers are the non-outlier correspondences (on-plane or off-plane).
  std::vector<char> true_inlier_mask;
};

// Generates a scene where `plane_fraction` of the true-inlier correspondences
// lie on a dominant plane and `outlier_fraction` of all correspondences are
// gross mismatches.
Scene GenerateScene(size_t num_points,
                    double plane_fraction,
                    double outlier_fraction,
                    double noise) {
  Scene scene;
  scene.K = RandomCalibrationMatrix();
  // A moderate forward-facing relative pose (limited rotation and a small
  // baseline relative to the scene depth) so that points stay in front of both
  // cameras and pose recovery is not affected by the twisted-pair ambiguity.
  const double angle = RandomUniformReal<double>(5.0, 30.0) * M_PI / 180.0;
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(angle, Eigen::Vector3d::Random().normalized()));
  const Eigen::Vector3d translation = Eigen::Vector3d::Random().normalized() *
                                      RandomUniformReal<double>(0.1, 0.4);
  scene.cam2_from_cam1 = Rigid3d(rotation, translation);
  const Eigen::Matrix3d K_inv = scene.K.inverse();
  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
  constexpr double kDistance = 2.0;

  const size_t num_outliers =
      static_cast<size_t>(std::round(outlier_fraction * num_points));
  const size_t num_inliers = num_points - num_outliers;
  const size_t num_on_plane =
      static_cast<size_t>(std::round(plane_fraction * num_inliers));

  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector2d point1 =
        scene.K.topRows<2>() * Eigen::Vector2d::Random().homogeneous();
    const Eigen::Vector3d ray = K_inv * point1.homogeneous();

    const bool is_outlier = i >= num_inliers;
    const bool on_plane = !is_outlier && i < num_on_plane;

    double depth;
    if (on_plane) {
      depth = kDistance / normal.dot(ray);
    } else {
      depth = RandomUniformReal<double>(0.5, 3.0);
    }
    const Eigen::Vector3d point3D_in_cam1 = depth * ray;
    Eigen::Vector2d point2 =
        (scene.K * (scene.cam2_from_cam1 * point3D_in_cam1)).hnormalized();
    if (is_outlier) {
      // Replace with a random mismatch somewhere in the image.
      point2 = scene.K.topRows<2>() * Eigen::Vector2d::Random().homogeneous();
    }

    scene.points1.push_back(point1 + noise * Eigen::Vector2d::Random());
    scene.points2.push_back(point2 + noise * Eigen::Vector2d::Random());
    scene.true_inlier_mask.push_back(!is_outlier);
  }
  return scene;
}

double RotationErrorDeg(const Eigen::Matrix3d& R_gt, const Eigen::Matrix3d& R) {
  const Eigen::Quaterniond q_gt(R_gt);
  const Eigen::Quaterniond q(R);
  return q_gt.angularDistance(q) * 180.0 / M_PI;
}

double TranslationErrorDeg(const Eigen::Vector3d& t_gt,
                           const Eigen::Vector3d& t) {
  const double cos_angle =
      std::clamp(std::abs(t_gt.normalized().dot(t.normalized())), 0.0, 1.0);
  return std::acos(cos_angle) * 180.0 / M_PI;
}

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double rank = percentile / 100.0 * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  const double frac = rank - lo;
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0;
  for (double v : values) {
    sum += v;
  }
  return sum / values.size();
}

struct Stats {
  double success_rate = 0;  // Pose within 5 deg of ground truth.
  double recall = 0;        // % of true inliers recovered as inliers.
  double precision = 0;     // % of reported inliers that are true inliers.
  double rot_med = 0;
  double rot_p90 = 0;
  double trans_med = 0;
  double trans_p90 = 0;
  double time_mean_ms = 0;
  double time_p90_ms = 0;
  double avg_trials = 0;
};

// Recovers the relative pose from a fundamental matrix and returns rotation and
// translation-direction error in degrees.
void PoseErrors(const Scene& scene,
                const Eigen::Matrix3d& F,
                double* rot_err,
                double* trans_err) {
  const Eigen::Matrix3d E = EssentialFromFundamentalMatrix(scene.K, F, scene.K);
  std::vector<Eigen::Vector3d> rays1(scene.points1.size());
  std::vector<Eigen::Vector3d> rays2(scene.points2.size());
  const Eigen::Matrix3d K_inv = scene.K.inverse();
  for (size_t i = 0; i < scene.points1.size(); ++i) {
    rays1[i] = (K_inv * scene.points1[i].homogeneous()).normalized();
    rays2[i] = (K_inv * scene.points2[i].homogeneous()).normalized();
  }
  Rigid3d cam2_from_cam1;
  std::vector<int> valid;
  PoseFromEssentialMatrix(E, rays1, rays2, &cam2_from_cam1, &valid);
  *rot_err =
      RotationErrorDeg(scene.cam2_from_cam1.rotation().toRotationMatrix(),
                       cam2_from_cam1.rotation().toRotationMatrix());
  *trans_err = TranslationErrorDeg(scene.cam2_from_cam1.translation(),
                                   cam2_from_cam1.translation());
}

// Inlier recall (fraction of true inliers reported) and precision (fraction of
// reported inliers that are true inliers) for the estimated inlier mask.
void RecallPrecision(const Scene& scene,
                     const std::vector<char>& inlier_mask,
                     double* recall,
                     double* precision) {
  size_t num_true = 0;
  size_t num_reported = 0;
  size_t num_true_reported = 0;
  for (size_t i = 0; i < scene.true_inlier_mask.size(); ++i) {
    const bool is_true = scene.true_inlier_mask[i];
    const bool is_reported = i < inlier_mask.size() && inlier_mask[i];
    num_true += is_true;
    num_reported += is_reported;
    num_true_reported += is_true && is_reported;
  }
  *recall = num_true == 0 ? 0.0 : 100.0 * num_true_reported / num_true;
  *precision =
      num_reported == 0 ? 0.0 : 100.0 * num_true_reported / num_reported;
}

enum class Method { kLoRansac, kDegensac };

Stats RunConfig(Method method,
                size_t num_points,
                double plane_fraction,
                double outlier_fraction,
                double noise,
                int num_problems) {
  int num_success = 0;
  std::vector<double> rot_errs;
  std::vector<double> trans_errs;
  std::vector<double> times_ms;
  std::vector<double> recalls;
  std::vector<double> precisions;
  double sum_trials = 0;

  for (int p = 0; p < num_problems; ++p) {
    // Deterministic, distinct data per problem; identical for both methods.
    SetPRNGSeed(1000 + p);
    const Scene scene =
        GenerateScene(num_points, plane_fraction, outlier_fraction, noise);

    RANSACOptions ransac_options;
    ransac_options.max_error = kMaxError;
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.1;
    ransac_options.max_num_trials = 10000;
    ransac_options.random_seed = 5000 + p;

    LORANSAC<FundamentalMatrixSevenPointEstimator,
             FundamentalMatrixEightPointEstimator>::Report report;
    const auto start = std::chrono::high_resolution_clock::now();
    if (method == Method::kDegensac) {
      FundamentalMatrixDegensacOptions options;
      options.ransac = ransac_options;
      const auto r = EstimateFundamentalMatrixDegensac(
          scene.points1, scene.points2, options);
      report.success = r.success;
      report.num_trials = r.num_trials;
      report.support = r.support;
      report.inlier_mask = r.inlier_mask;
      report.model = r.model;
    } else {
      LORANSAC<FundamentalMatrixSevenPointEstimator,
               FundamentalMatrixEightPointEstimator>
          loransac(ransac_options);
      report = loransac.Estimate(scene.points1, scene.points2);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    times_ms.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());

    sum_trials += report.num_trials;
    if (!report.success) {
      // A failed estimate counts as a full pose failure with worst-case error.
      rot_errs.push_back(180.0);
      trans_errs.push_back(90.0);
      recalls.push_back(0.0);
      precisions.push_back(0.0);
      continue;
    }

    double rot_err;
    double trans_err;
    PoseErrors(scene, report.model, &rot_err, &trans_err);
    if (rot_err < 5.0 && trans_err < 5.0) {
      ++num_success;
    }
    rot_errs.push_back(rot_err);
    trans_errs.push_back(trans_err);

    double recall;
    double precision;
    RecallPrecision(scene, report.inlier_mask, &recall, &precision);
    recalls.push_back(recall);
    precisions.push_back(precision);
  }

  Stats stats;
  stats.success_rate = 100.0 * num_success / num_problems;
  stats.recall = Mean(recalls);
  stats.precision = Mean(precisions);
  stats.rot_med = Percentile(rot_errs, 50);
  stats.rot_p90 = Percentile(rot_errs, 90);
  stats.trans_med = Percentile(trans_errs, 50);
  stats.trans_p90 = Percentile(trans_errs, 90);
  stats.time_mean_ms = Mean(times_ms);
  stats.time_p90_ms = Percentile(times_ms, 90);
  stats.avg_trials = sum_trials / num_problems;
  return stats;
}

void PrintRow(size_t num_points,
              double plane_fraction,
              double outlier_fraction,
              const char* method,
              const Stats& stats) {
  std::printf(
      "| %5zu | %5.0f%% | %4.0f%% | %-8s | %7.2f | %7.2f | %6.1f | %8.1f | "
      "%7.1f | %8.3f | %8.3f | %9.3f | %9.3f | %6.0f |\n",
      num_points,
      100 * plane_fraction,
      100 * outlier_fraction,
      method,
      stats.time_mean_ms,
      stats.time_p90_ms,
      stats.success_rate,
      stats.recall,
      stats.precision,
      stats.rot_med,
      stats.rot_p90,
      stats.trans_med,
      stats.trans_p90,
      stats.avg_trials);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  constexpr int kNumProblems = 300;
  constexpr double kNoise = 0.5;  // Pixel std-dev of observation noise.
  const std::vector<size_t> point_counts = {200, 1000};
  const std::vector<double> plane_fractions = {
      0.0, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98};
  const std::vector<double> outlier_fractions = {0.0, 0.2, 0.4};

  std::printf(
      "DEGENSAC vs plain LO-RANSAC for fundamental matrix estimation on "
      "synthetic\ntwo-view data with a dominant plane.\n\n");
  std::printf(
      "- %d independent problems per configuration; both methods see identical "
      "data\n  and identical RANSAC randomness per problem.\n"
      "- Observation noise: %.1f px std-dev; inlier threshold max_error=%.1f "
      "px.\n"
      "- `success` = %% of runs whose recovered pose is within 5 deg of ground "
      "truth.\n"
      "- `recall` = %% of true inliers recovered; `prec` = %% of reported "
      "inliers that\n  are true inliers (both averaged over all problems, "
      "failures included).\n"
      "- `rot`/`trans` errors (deg) are over ALL returned models (a "
      "plane-corrupted\n  model contributes its large error), so medians and "
      "p90 reflect degradation.\n\n",
      kNumProblems,
      kNoise,
      kMaxError);
  std::printf(
      "| Pts   | Plane  | Outl | Method   | t mean  | t p90   | Succ%%  | "
      "Recall%% | Prec%%  | Rot med  | Rot p90  | Trans med | Trans p90 | "
      "Trials |\n");
  std::printf(
      "|------:|-------:|-----:|:---------|--------:|--------:|-------:|"
      "--------:|-------:|---------:|---------:|----------:|----------:|"
      "-------:|\n");

  for (size_t num_points : point_counts) {
    for (double plane_fraction : plane_fractions) {
      for (double outlier_fraction : outlier_fractions) {
        const Stats loransac_stats = RunConfig(Method::kLoRansac,
                                               num_points,
                                               plane_fraction,
                                               outlier_fraction,
                                               kNoise,
                                               kNumProblems);
        const Stats degensac_stats = RunConfig(Method::kDegensac,
                                               num_points,
                                               plane_fraction,
                                               outlier_fraction,
                                               kNoise,
                                               kNumProblems);
        PrintRow(num_points,
                 plane_fraction,
                 outlier_fraction,
                 "loransac",
                 loransac_stats);
        PrintRow(num_points,
                 plane_fraction,
                 outlier_fraction,
                 "degensac",
                 degensac_stats);
      }
    }
  }

  return 0;
}
