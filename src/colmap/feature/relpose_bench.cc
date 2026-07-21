// TEMPORARY (uncommitted) micro-benchmark: relative-pose accuracy + runtime of
// the essential-matrix RANSAC, sweeping the local-optimizer (LO) axis while
// holding the tangent-Sampson scoring fixed. Configs B/C/D/E live in this one
// binary (same seed, same synthetic scenes) so they are directly comparable;
// config A (main's bearing scoring) is measured by the twin bench in
// ../colmap_base. Remove before merge (as with spherical_bench.cc).
//
//   C = tangent scoring, 5-point-refit LO (no Refine).
//   B = tangent scoring, buggy bearing-Sampson LM (Sampson on unit bearings).
//   D = tangent scoring, autodiff tangent-Sampson LM.
//   E = tangent scoring, analytic tangent-Sampson LM (current production).

#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/scene/camera.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/tiny_solver_autodiff_function.h>

using namespace colmap;

namespace {

using RelativePoseManifold =
    ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;

// ---------------------------------------------------------------------------
// Autodiff LM refines (configs B and D): batched tiny functors over all
// inliers.
// ---------------------------------------------------------------------------
struct BearingSampsonBatch {
  const std::vector<CamRayWithJac>* x;
  const std::vector<CamRayWithJac>* y;
  int NumResiduals() const { return static_cast<int>(x->size()); }
  template <typename T>
  bool operator()(const T* p, T* res) const {
    const Eigen::Matrix<T, 3, 3> E = EssentialMatrixFromPoseParams(p);
    for (size_t i = 0; i < x->size(); ++i) {
      res[i] = SampsonError<T>(E, (*x)[i].ray.cast<T>(), (*y)[i].ray.cast<T>());
    }
    return true;
  }
};

struct TangentSampsonBatch {
  const std::vector<CamRayWithJac>* x;
  const std::vector<CamRayWithJac>* y;
  int NumResiduals() const { return static_cast<int>(x->size()); }
  template <typename T>
  bool operator()(const T* p, T* res) const {
    const Eigen::Matrix<T, 3, 3> E = EssentialMatrixFromPoseParams(p);
    for (size_t i = 0; i < x->size(); ++i) {
      res[i] = TangentSampsonError<T>(E,
                                      (*x)[i].ray.cast<T>(),
                                      (*x)[i].jacobian.cast<T>(),
                                      (*y)[i].ray.cast<T>(),
                                      (*y)[i].jacobian.cast<T>());
    }
    return true;
  }
};

template <typename Batch>
bool RefineAutodiff(const std::vector<CamRayWithJac>& x,
                    const std::vector<CamRayWithJac>& y,
                    Eigen::Matrix3d* E) {
  std::vector<Eigen::Vector3d> r1(x.size()), r2(y.size());
  for (size_t i = 0; i < x.size(); ++i) {
    r1[i] = x[i].ray;
    r2[i] = y[i].ray;
  }
  Rigid3d pose;
  std::vector<int> valid;
  PoseFromEssentialMatrix(*E, r1, r2, &pose, &valid);
  if (valid.empty()) return false;

  Batch batch{&x, &y};
  ceres::TinySolverAutoDiffFunction<Batch, Eigen::Dynamic, 7> f(batch);
  TinySolver<decltype(f), RelativePoseManifold> solver;
  typename TinySolver<decltype(f), RelativePoseManifold>::Options options;
  options.max_num_iterations = 25;
  Eigen::Matrix<double, 7, 1> xx;
  xx.head<4>() = pose.rotation().normalized().coeffs();
  xx.tail<3>() = pose.translation().normalized();
  solver.Solve(f, &xx, options);
  if (xx.allFinite()) {
    pose = Rigid3d(Eigen::Quaterniond(xx.data()).normalized(), xx.tail<3>());
  }
  *E = EssentialMatrixFromPose(pose);
  return true;
}

// ---------------------------------------------------------------------------
// Four wrapper estimators. All share tangent scoring (Estimate + Residuals);
// they differ only in the LO the LORANSAC trait picks up (via Refine or not).
// ---------------------------------------------------------------------------
struct EstBase {
  using X_t = CamRayWithJac;
  using Y_t = CamRayWithJac;
  using M_t = Eigen::Matrix3d;
  static constexpr int kMinNumSamples = 5;
  static void Estimate(const std::vector<X_t>& x,
                       const std::vector<Y_t>& y,
                       std::vector<M_t>* m) {
    EssentialMatrixTangentSampsonEstimator::Estimate(x, y, m);
  }
  static void Residuals(const std::vector<X_t>& x,
                        const std::vector<Y_t>& y,
                        const M_t& E,
                        std::vector<double>* r) {
    EssentialMatrixTangentSampsonEstimator::Residuals(x, y, E, r);
  }
};

struct EstC : EstBase {};  // no Refine -> 5-point-refit LO

struct EstB : EstBase {
  static bool Refine(const std::vector<X_t>& x,
                     const std::vector<Y_t>& y,
                     M_t* E) {
    return RefineAutodiff<BearingSampsonBatch>(x, y, E);
  }
};

struct EstD : EstBase {
  static bool Refine(const std::vector<X_t>& x,
                     const std::vector<Y_t>& y,
                     M_t* E) {
    return RefineAutodiff<TangentSampsonBatch>(x, y, E);
  }
};

struct EstE : EstBase {
  static bool Refine(const std::vector<X_t>& x,
                     const std::vector<Y_t>& y,
                     M_t* E) {
    return EssentialMatrixTangentSampsonEstimator::Refine(x, y, E);
  }
};

// ---------------------------------------------------------------------------
// Synthetic scenes. IDENTICAL generation must be mirrored in the base bench so
// both worktrees see the same data (SetPRNGSeed(0), same draw order).
// ---------------------------------------------------------------------------
struct Trial {
  Rigid3d gt;
  std::vector<Eigen::Vector2d> px1, px2;
};

std::vector<Trial> GenerateTrials(const Camera& cam,
                                  bool spherical,
                                  double outlier_frac,
                                  int num_trials) {
  SetPRNGSeed(0);
  constexpr double kNoisePx = 1.0;
  constexpr int kMaxPoints = 200;
  std::vector<Trial> trials;
  trials.reserve(num_trials);
  for (int t = 0; t < num_trials; ++t) {
    Trial trial;
    trial.gt = Rigid3d(Eigen::Quaterniond(RandomEigenVectord<4>().normalized()),
                       RandomEigenVectord<3>().normalized());
    int attempts = 0;
    while (static_cast<int>(trial.px1.size()) < kMaxPoints && attempts < 2000) {
      ++attempts;
      Eigen::Vector3d p1;
      if (spherical) {
        p1 = RandomEigenVectord<3>().normalized() *
             RandomUniformReal<double>(0.5, 4.0);
      } else {
        p1 = Eigen::Vector3d(RandomUniformReal<double>(-1.0, 1.0),
                             RandomUniformReal<double>(-1.0, 1.0),
                             RandomUniformReal<double>(0.5, 4.0));
      }
      const Eigen::Vector3d p2 = trial.gt * p1;
      if (!spherical && p2.z() <= 0.1) continue;
      const auto img1 = cam.ImgFromCam(p1);
      const auto img2 = cam.ImgFromCam(p2);
      if (!img1 || !img2) continue;
      Eigen::Vector2d q1 =
          *img1 + kNoisePx * Eigen::Vector2d(RandomUniformReal<double>(-1, 1),
                                             RandomUniformReal<double>(-1, 1));
      Eigen::Vector2d q2;
      if (RandomUniformReal<double>(0.0, 1.0) < outlier_frac) {
        q2 = Eigen::Vector2d(RandomUniformReal<double>(0.0, cam.width),
                             RandomUniformReal<double>(0.0, cam.height));
      } else {
        q2 = *img2 +
             kNoisePx * Eigen::Vector2d(RandomUniformReal<double>(-1, 1),
                                        RandomUniformReal<double>(-1, 1));
      }
      trial.px1.push_back(q1);
      trial.px2.push_back(q2);
    }
    trials.push_back(std::move(trial));
  }
  return trials;
}

double Median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double RotErrDeg(const Rigid3d& a, const Rigid3d& b) {
  return Eigen::AngleAxisd(Eigen::Quaterniond(a.rotation()) *
                           Eigen::Quaterniond(b.rotation()).inverse())
             .angle() *
         180.0 / M_PI;
}

double TransErrDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)) *
         180.0 / M_PI;
}

template <typename Est>
void RunConfig(const char* name,
               const Camera& cam,
               const std::vector<Trial>& trials) {
  RANSACOptions options;
  options.max_error = 2.0;  // pixels (tangent Sampson)
  options.min_inlier_ratio = 0.1;
  options.confidence = 0.9999;
  options.min_num_trials = 100;
  options.max_num_trials =
      100;  // fixed iteration budget for a fair LO comparison

  // Re-seed so every config draws the SAME RANSAC minimal samples (100 fixed
  // iterations each), isolating the LO as the only difference across B/C/D/E.
  SetPRNGSeed(0);
  std::vector<double> ms, rot, trans;
  ms.reserve(trials.size());
  for (const Trial& trial : trials) {
    std::vector<CamRayWithJac> X, Y;
    X.reserve(trial.px1.size());
    Y.reserve(trial.px2.size());
    bool ok = true;
    for (size_t i = 0; i < trial.px1.size(); ++i) {
      auto rx = cam.CamRayFromImgWithJac(trial.px1[i]);
      auto ry = cam.CamRayFromImgWithJac(trial.px2[i]);
      if (!rx || !ry) {
        ok = false;
        break;
      }
      X.push_back(*rx);
      Y.push_back(*ry);
    }
    if (!ok || X.size() < 5) continue;

    LORANSAC<Est, Est> ransac(options);
    const auto t0 = std::chrono::steady_clock::now();
    const auto report = ransac.Estimate(X, Y);
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (!report.success) continue;

    std::vector<Eigen::Vector3d> in1, in2;
    for (size_t i = 0; i < X.size(); ++i) {
      if (report.inlier_mask[i]) {
        in1.push_back(X[i].ray);
        in2.push_back(Y[i].ray);
      }
    }
    Rigid3d est;
    std::vector<int> valid;
    PoseFromEssentialMatrix(report.model, in1, in2, &est, &valid);
    if (valid.empty()) continue;
    rot.push_back(RotErrDeg(est, trial.gt));
    trans.push_back(TransErrDeg(Eigen::Vector3d(trial.gt.translation()),
                                Eigen::Vector3d(est.translation())));
  }
  std::printf("    %-2s  rot=%.4f  trans=%.4f deg   time median=%.3f ms\n",
              name,
              Median(rot),
              Median(trans),
              Median(ms));
}

void RunCamera(const char* cam_name, const Camera& cam, bool spherical) {
  for (const double outl : {0.0, 0.3}) {
    const auto trials = GenerateTrials(cam, spherical, outl, 1000);
    std::printf("  %s, outliers=%.0f%%:\n", cam_name, outl * 100);
    RunConfig<EstC>("C", cam, trials);
    RunConfig<EstB>("B", cam, trials);
    RunConfig<EstD>("D", cam, trials);
    RunConfig<EstE>("E", cam, trials);
  }
}

}  // namespace

int main() {
  Camera pinhole =
      Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 800.0, 1024, 768);
  Camera opencv = Camera::CreateFromModelName(2, "OPENCV", 300.0, 1024, 768);
  opencv.params[4] = -0.28;  // k1
  opencv.params[5] = 0.07;   // k2
  Camera equirect =
      Camera::CreateFromModelName(3, "EQUIRECTANGULAR", 1.0, 1000, 500);

  std::printf(
      "Configs B/C/D/E (tangent scoring; LO varies), 1000 trials, 1px noise\n");
  RunCamera("SimplePinhole", pinhole, false);
  RunCamera("OpenCV(wide+dist)", opencv, false);
  RunCamera("Equirectangular", equirect, true);
  return 0;
}
