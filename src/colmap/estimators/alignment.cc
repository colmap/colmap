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

#include "colmap/estimators/alignment.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/solvers/similarity_transform.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/math.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/projection.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <memory>

#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>

namespace colmap {
namespace {

struct ReconstructionAlignmentEstimator {
  static const int kMinNumSamples = 3;

  using X_t = const Image*;
  using Y_t = const Image*;
  using M_t = Sim3d;

  ReconstructionAlignmentEstimator(double max_reproj_error,
                                   const Reconstruction* src_reconstruction,
                                   const Reconstruction* tgt_reconstruction)
      : max_squared_reproj_error_(max_reproj_error * max_reproj_error),
        src_reconstruction_(src_reconstruction),
        tgt_reconstruction_(tgt_reconstruction) {
    THROW_CHECK_GE(max_reproj_error, 0);
    THROW_CHECK_NOTNULL(src_reconstruction_);
    THROW_CHECK_NOTNULL(tgt_reconstruction_);
  }

  // Estimate 3D similarity transform from corresponding projection centers.
  void Estimate(const std::vector<X_t>& src_images,
                const std::vector<Y_t>& tgt_images,
                std::vector<M_t>* models) const {
    THROW_CHECK_GE(src_images.size(), 3);
    THROW_CHECK_GE(tgt_images.size(), 3);
    THROW_CHECK_EQ(src_images.size(), tgt_images.size());
    THROW_CHECK(models != nullptr);

    models->clear();

    std::vector<Eigen::Vector3d> proj_centers1(src_images.size());
    std::vector<Eigen::Vector3d> proj_centers2(tgt_images.size());
    for (size_t i = 0; i < src_images.size(); ++i) {
      THROW_CHECK_EQ(src_images[i]->ImageId(), tgt_images[i]->ImageId());
      proj_centers1[i] = src_images[i]->ProjectionCenter();
      proj_centers2[i] = tgt_images[i]->ProjectionCenter();
    }

    Sim3d tgt_from_src;
    if (!EstimateSim3d(proj_centers1, proj_centers2, tgt_from_src)) {
      return;
    }

    models->resize(1);
    (*models)[0] = tgt_from_src;
  }

  // For each image, determine the ratio of 3D points that correctly project
  // from one image to the other image and vice versa for the given
  // tgt_from_src. The residual is then defined as 1 minus this ratio, i.e., an
  // error threshold of 0.3 means that 70% of the points for that image must
  // reproject within the given maximum reprojection error threshold.
  void Residuals(const std::vector<X_t>& src_images,
                 const std::vector<Y_t>& tgt_images,
                 const M_t& tgt_from_src,
                 std::vector<double>* residuals) const {
    THROW_CHECK_EQ(src_images.size(), tgt_images.size());
    THROW_CHECK_NOTNULL(src_reconstruction_);
    THROW_CHECK_NOTNULL(tgt_reconstruction_);

    const Sim3d src_from_tgt = Inverse(tgt_from_src);

    residuals->resize(src_images.size());

    for (size_t i = 0; i < src_images.size(); ++i) {
      const Image& src_image = *src_images[i];
      const Image& tgt_image = *tgt_images[i];

      THROW_CHECK_EQ(src_image.ImageId(), tgt_image.ImageId());

      const Camera& src_camera = *src_image.CameraPtr();
      const Camera& tgt_camera = *tgt_image.CameraPtr();

      const Eigen::Matrix3x4d src_cam_from_world =
          src_image.CamFromWorld().ToMatrix();
      const Eigen::Matrix3x4d tgt_cam_from_world =
          tgt_image.CamFromWorld().ToMatrix();

      THROW_CHECK_EQ(src_image.NumPoints2D(), tgt_image.NumPoints2D());

      size_t num_inliers = 0;
      size_t num_common_points = 0;

      for (point2D_t point2D_idx = 0; point2D_idx < src_image.NumPoints2D();
           ++point2D_idx) {
        // Check if both images have a 3D point.

        const auto& src_point2D = src_image.Point2D(point2D_idx);
        if (!src_point2D.HasPoint3D()) {
          continue;
        }

        const auto& tgt_point2D = tgt_image.Point2D(point2D_idx);
        if (!tgt_point2D.HasPoint3D()) {
          continue;
        }

        num_common_points += 1;

        const Eigen::Vector3d src_point_in_tgt =
            tgt_from_src *
            src_reconstruction_->Point3D(src_point2D.point3D_id).xyz;
        if (CalculateSquaredReprojectionError(tgt_point2D.xy,
                                              src_point_in_tgt,
                                              tgt_cam_from_world,
                                              tgt_camera) >
            max_squared_reproj_error_) {
          continue;
        }

        const Eigen::Vector3d tgt_point_in_src =
            src_from_tgt *
            tgt_reconstruction_->Point3D(tgt_point2D.point3D_id).xyz;
        if (CalculateSquaredReprojectionError(src_point2D.xy,
                                              tgt_point_in_src,
                                              src_cam_from_world,
                                              src_camera) >
            max_squared_reproj_error_) {
          continue;
        }

        num_inliers += 1;
      }

      if (num_common_points == 0) {
        (*residuals)[i] = 1.0;
      } else {
        const double negative_inlier_ratio =
            1.0 - static_cast<double>(num_inliers) /
                      static_cast<double>(num_common_points);
        (*residuals)[i] = negative_inlier_ratio * negative_inlier_ratio;
      }
    }
  }

 private:
  double max_squared_reproj_error_;
  const Reconstruction* src_reconstruction_;
  const Reconstruction* tgt_reconstruction_;
};

constexpr double kChiSquare95ThreeDofSqrt = 2.7955;

Eigen::Matrix3d SqrtInformationOrFallback(const Eigen::Matrix3d& covariance,
                                          const double fallback_stddev) {
  THROW_CHECK_GT(fallback_stddev, 0.0);
  if (covariance.allFinite()) {
    const Eigen::Matrix3d symmetric =
        0.5 * (covariance + covariance.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(symmetric);
    if (eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
        eig.eigenvalues().minCoeff() > 1e-12) {
      return eig.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
             eig.eigenvectors().transpose();
    }
  }
  return Eigen::Matrix3d::Identity() / fallback_stddev;
}

struct AlignmentPositionCostFunctor {
  AlignmentPositionCostFunctor(const Eigen::Vector3d& center_in_src,
                               const Eigen::Vector3d& center_in_tgt,
                               const Eigen::Matrix3d& sqrt_information)
      : center_in_src_(center_in_src),
        center_in_tgt_(center_in_tgt),
        sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const tgt_from_src_rotation,
                  const T* const tgt_from_src_translation,
                  const T* const tgt_from_src_scale,
                  T* residuals_ptr) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(
        tgt_from_src_rotation);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(
        tgt_from_src_translation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = sqrt_information_.cast<T>() *
                (tgt_from_src_scale[0] * (rotation * center_in_src_.cast<T>()) +
                 translation - center_in_tgt_.cast<T>());
    return true;
  }

  const Eigen::Vector3d center_in_src_;
  const Eigen::Vector3d center_in_tgt_;
  const Eigen::Matrix3d sqrt_information_;
};

struct AlignmentOrientationCostFunctor {
  AlignmentOrientationCostFunctor(
      const Eigen::Quaterniond& sensor_from_src,
      const Eigen::Quaterniond& sensor_from_tgt_measured,
      const Eigen::Matrix3d& sqrt_information)
      : sensor_from_src_(sensor_from_src),
        tgt_from_sensor_measured_(sensor_from_tgt_measured.inverse()),
        sqrt_information_(sqrt_information) {}

  template <typename T>
  bool operator()(const T* const tgt_from_src_rotation,
                  T* residuals_ptr) const {
    const Eigen::Map<const Eigen::Quaternion<T>> tgt_from_src(
        tgt_from_src_rotation);
    const Eigen::Quaternion<T> sensor_from_tgt_predicted =
        sensor_from_src_.cast<T>() * tgt_from_src.inverse();
    const Eigen::Quaternion<T> measured_from_predicted =
        tgt_from_sensor_measured_.cast<T>() * sensor_from_tgt_predicted;
    Eigen::Matrix<T, 3, 1> angle_axis;
    AngleAxisFromEigenQuaternion(measured_from_predicted.coeffs().data(),
                                 angle_axis.data());
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = sqrt_information_.cast<T>() * angle_axis;
    return true;
  }

  const Eigen::Quaterniond sensor_from_src_;
  const Eigen::Quaterniond tgt_from_sensor_measured_;
  const Eigen::Matrix3d sqrt_information_;
};

// Anisotropic ENU horizontal/vertical position RANSAC gate (workstream 4A).
// Model estimation is identical to SimilarityTransformEstimator<3, true>
// (delegated directly, so the estimated Sim3d is exactly the same equal-
// weight Umeyama fit); only the residual/inlier test differs. Assumes both
// src and tgt points are already expressed in a shared ENU frame (Up = +Z):
// the residual is normalized so that comparing it against
// RANSACOptions.max_error == max_horizontal_error (the generic RANSAC
// template squares max_error into `max_residual`) is equivalent to
// requiring horizontal_error <= max_horizontal_error AND
// vertical_error <= max_vertical_error independently.
struct AnisotropicEnuPositionEstimator {
  static const int kMinNumSamples = 3;

  using X_t = Eigen::Vector3d;
  using Y_t = Eigen::Vector3d;
  using M_t = Eigen::Matrix3x4d;

  AnisotropicEnuPositionEstimator(double max_horizontal_error,
                                  double max_vertical_error)
      : max_horizontal_error_(max_horizontal_error),
        max_vertical_error_(max_vertical_error) {
    THROW_CHECK_GT(max_horizontal_error, 0.0);
    THROW_CHECK_GT(max_vertical_error, 0.0);
  }

  void Estimate(const std::vector<X_t>& src,
               const std::vector<Y_t>& tgt,
               std::vector<M_t>* models) const {
    SimilarityTransformEstimator<3, true>::Estimate(src, tgt, models);
  }

  void Residuals(const std::vector<X_t>& src,
                const std::vector<Y_t>& tgt,
                const M_t& tgt_from_src,
                std::vector<double>* residuals) const {
    const size_t num_points = src.size();
    THROW_CHECK_EQ(num_points, tgt.size());
    residuals->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const Eigen::Vector3d src_transformed =
          tgt_from_src * src[i].homogeneous();
      const Eigen::Vector3d diff = tgt[i] - src_transformed;
      const double horizontal_error = std::hypot(diff.x(), diff.y());
      const double vertical_error = std::abs(diff.z());
      const double normalized =
          std::max(horizontal_error / max_horizontal_error_,
                   vertical_error / max_vertical_error_);
      const double scaled = normalized * max_horizontal_error_;
      (*residuals)[i] = scaled * scaled;
    }
  }

 private:
  double max_horizontal_error_;
  double max_vertical_error_;
};

double OrientationResidualDeg(const Image& image,
                              const PosePrior& prior,
                              const Sim3d& tgt_from_src) {
  const Eigen::Quaterniond sensor_from_tgt_predicted =
      image.CamFromWorld().rotation() * tgt_from_src.rotation().inverse();
  return RadToDeg(prior.rotation.angularDistance(sensor_from_tgt_predicted));
}

bool SolveJointPosePriorAlignment(
    const Reconstruction& src_reconstruction,
    const NodeHashMap<image_t, const PosePrior*>& priors_by_image,
    const std::vector<image_t>& position_image_ids,
    const std::vector<image_t>& orientation_image_ids,
    const double position_fallback_stddev,
    const double orientation_fallback_stddev_rad,
    Sim3d* tgt_from_src) {
  Eigen::Quaterniond rotation = tgt_from_src->rotation();
  Eigen::Vector3d translation = tgt_from_src->translation();
  double scale = tgt_from_src->scale();

  ceres::Problem problem;
  problem.AddParameterBlock(rotation.coeffs().data(), 4);
  SetManifold(
      &problem, rotation.coeffs().data(), CreateEigenQuaternionManifold());
  problem.AddParameterBlock(translation.data(), 3);
  problem.AddParameterBlock(&scale, 1);
  problem.SetParameterLowerBound(&scale, 0, 1e-12);

  auto* position_loss = new ceres::CauchyLoss(kChiSquare95ThreeDofSqrt);
  for (const image_t image_id : position_image_ids) {
    const PosePrior& prior = *priors_by_image.at(image_id);
    const Eigen::Matrix3d sqrt_information = SqrtInformationOrFallback(
        prior.position_covariance, position_fallback_stddev);
    problem.AddResidualBlock(
        new ceres::
            AutoDiffCostFunction<AlignmentPositionCostFunctor, 3, 4, 3, 1>(
                new AlignmentPositionCostFunctor(
                    src_reconstruction.Image(image_id).ProjectionCenter(),
                    prior.position,
                    sqrt_information)),
        position_loss,
        rotation.coeffs().data(),
        translation.data(),
        &scale);
  }

  auto* orientation_loss = new ceres::CauchyLoss(kChiSquare95ThreeDofSqrt);
  for (const image_t image_id : orientation_image_ids) {
    const PosePrior& prior = *priors_by_image.at(image_id);
    const Eigen::Matrix3d sqrt_information = SqrtInformationOrFallback(
        prior.rotation_covariance, orientation_fallback_stddev_rad);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<AlignmentOrientationCostFunctor, 3, 4>(
            new AlignmentOrientationCostFunctor(
                src_reconstruction.Image(image_id).CamFromWorld().rotation(),
                prior.rotation,
                sqrt_information)),
        orientation_loss,
        rotation.coeffs().data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.function_tolerance = 1e-10;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (!summary.IsSolutionUsable() || !rotation.coeffs().allFinite() ||
      !translation.allFinite() || !std::isfinite(scale) || scale <= 0.0) {
    return false;
  }

  *tgt_from_src = Sim3d(scale, rotation.normalized(), translation);
  return true;
}

}  // namespace

bool AlignReconstructionToLocations(
    const Reconstruction& src_reconstruction,
    const std::vector<std::string>& tgt_image_names,
    const std::vector<Eigen::Vector3d>& tgt_image_locations,
    const int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GE(min_common_images, 3);
  THROW_CHECK_EQ(tgt_image_names.size(), tgt_image_locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  FlatHashSet<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < tgt_image_names.size(); ++i) {
    const class Image* src_image =
        src_reconstruction.FindImageWithName(tgt_image_names[i]);
    if (src_image == nullptr) {
      continue;
    }

    if (!src_image->HasPose()) {
      continue;
    }

    // Ignore duplicate images.
    if (!common_image_ids.insert(src_image->ImageId()).second) {
      continue;
    }

    src.push_back(src_image->ProjectionCenter());
    dst.push_back(tgt_image_locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  Sim3d tgt_from_src_;
  const auto report =
      EstimateSim3dRobust(src, dst, ransac_options, tgt_from_src_);

  if (report.support.num_inliers < static_cast<size_t>(min_common_images)) {
    return false;
  }

  if (tgt_from_src != nullptr) {
    *tgt_from_src = tgt_from_src_;
  }

  return true;
}

bool AlignReconstructionToPosePriors(
    const Reconstruction& src_reconstruction,
    const std::vector<PosePrior>& tgt_pose_priors,
    const RANSACOptions& ransac_options,
    Sim3d* tgt_from_src) {
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  src.reserve(tgt_pose_priors.size());
  tgt.reserve(tgt_pose_priors.size());

  NodeHashMap<image_t, PosePrior> tgt_image_to_pose_prior;
  for (const auto& pose_prior : tgt_pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        pose_prior.HasPosition()) {
      THROW_CHECK(tgt_image_to_pose_prior
                      .emplace(pose_prior.corr_data_id.id, pose_prior)
                      .second)
          << "Duplicate pose prior for image " << pose_prior.corr_data_id.id;
    }
  }

  for (const image_t image_id : src_reconstruction.RegImageIds()) {
    const auto pose_prior_it = tgt_image_to_pose_prior.find(image_id);
    if (pose_prior_it != tgt_image_to_pose_prior.end()) {
      const auto& image = src_reconstruction.Image(image_id);
      src.push_back(image.ProjectionCenter());
      tgt.push_back(pose_prior_it->second.position);
    }
  }

  if (src.size() < 3) {
    LOG(WARNING) << "Not enough valid pose priors for alignment";
    return false;
  }

  if (ransac_options.max_error > 0) {
    return EstimateSim3dRobust(src, tgt, ransac_options, *tgt_from_src).success;
  }
  return EstimateSim3d(src, tgt, *tgt_from_src);
}

PosePriorAlignmentResult AlignReconstructionToPosePriorsRobust(
    const Reconstruction& src_reconstruction,
    const std::vector<PosePrior>& tgt_pose_priors,
    const RANSACOptions& ransac_options,
    const AnisotropicPositionGate& anisotropic_gate) {
  PosePriorAlignmentResult result;

  NodeHashMap<image_t, PosePrior> tgt_image_to_pose_prior;
  for (const auto& pose_prior : tgt_pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA &&
        pose_prior.HasPosition()) {
      THROW_CHECK(tgt_image_to_pose_prior
                      .emplace(pose_prior.corr_data_id.id, pose_prior)
                      .second)
          << "Duplicate pose prior for image " << pose_prior.corr_data_id.id;
    }
  }

  // Collect correspondences in lexicographic image-name order.
  std::vector<std::pair<std::string, image_t>> sorted_image_ids;
  for (const image_t image_id : src_reconstruction.RegImageIds()) {
    if (tgt_image_to_pose_prior.find(image_id) !=
        tgt_image_to_pose_prior.end()) {
      sorted_image_ids.emplace_back(src_reconstruction.Image(image_id).Name(),
                                    image_id);
    }
  }
  std::sort(sorted_image_ids.begin(), sorted_image_ids.end());

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  src.reserve(sorted_image_ids.size());
  tgt.reserve(sorted_image_ids.size());
  for (const auto& [name, image_id] : sorted_image_ids) {
    result.correspondence_image_ids.push_back(image_id);
    src.push_back(src_reconstruction.Image(image_id).ProjectionCenter());
    tgt.push_back(tgt_image_to_pose_prior.at(image_id).position);
  }

  if (src.size() < 3) {
    LOG(WARNING) << "Not enough valid pose priors for alignment";
    result.inlier_mask.assign(src.size(), 0);
    return result;
  }

  if (anisotropic_gate.IsSet()) {
    // The generic RANSAC template squares RANSACOptions.max_error into its
    // inlier threshold; AnisotropicEnuPositionEstimator::Residuals()
    // normalizes by max_horizontal_error so that threshold is meaningful.
    RANSACOptions gate_options = ransac_options;
    gate_options.max_error = anisotropic_gate.max_horizontal_error;
    RANSAC<AnisotropicEnuPositionEstimator> ransac(
        gate_options,
        AnisotropicEnuPositionEstimator(anisotropic_gate.max_horizontal_error,
                                        anisotropic_gate.max_vertical_error));
    const auto report = ransac.Estimate(src, tgt);
    result.success = report.success;
    result.inlier_mask = report.inlier_mask;
    if (result.success) {
      result.tgt_from_src = Sim3d::FromMatrix(report.model);
    }
    if (result.inlier_mask.empty()) {
      result.inlier_mask.assign(src.size(), result.success ? 1 : 0);
    }
    return result;
  }

  const auto report =
      EstimateSim3dRobust(src, tgt, ransac_options, result.tgt_from_src);
  result.success = report.success;
  result.inlier_mask = report.inlier_mask;
  if (result.inlier_mask.empty()) {
    result.inlier_mask.assign(src.size(), result.success ? 1 : 0);
  }
  return result;
}

void RefinePosePriorAlignmentWithOrientations(
    const Reconstruction& src_reconstruction,
    const std::vector<PosePrior>& tgt_pose_priors,
    const double position_fallback_stddev,
    const double orientation_fallback_stddev_rad,
    const double orientation_max_error_deg,
    PosePriorAlignmentResult* result) {
  THROW_CHECK_NOTNULL(result);
  THROW_CHECK(result->success);
  THROW_CHECK_GT(position_fallback_stddev, 0.0);
  THROW_CHECK_GT(orientation_fallback_stddev_rad, 0.0);
  THROW_CHECK_GT(orientation_max_error_deg, 0.0);

  result->orientation_requested = true;
  result->orientation_engaged = false;
  result->orientation_image_ids.clear();
  result->orientation_inlier_mask.clear();
  result->orientation_residuals_deg.clear();

  NodeHashMap<image_t, const PosePrior*> priors_by_image;
  for (const PosePrior& prior : tgt_pose_priors) {
    if (prior.corr_data_id.sensor_id.type != SensorType::CAMERA) {
      continue;
    }
    const image_t image_id = static_cast<image_t>(prior.corr_data_id.id);
    THROW_CHECK(priors_by_image.emplace(image_id, &prior).second)
        << "Duplicate pose prior for image " << image_id;
  }

  std::vector<image_t> position_image_ids;
  for (size_t i = 0; i < result->correspondence_image_ids.size(); ++i) {
    if (result->inlier_mask[i]) {
      position_image_ids.push_back(result->correspondence_image_ids[i]);
    }
  }
  if (position_image_ids.size() < 3) {
    return;
  }

  std::vector<std::pair<std::string, image_t>> sorted_orientation_images;
  for (const image_t image_id : src_reconstruction.RegImageIds()) {
    const auto prior_it = priors_by_image.find(image_id);
    if (prior_it != priors_by_image.end() && prior_it->second->HasRotation()) {
      sorted_orientation_images.emplace_back(
          src_reconstruction.Image(image_id).Name(), image_id);
    }
  }
  std::sort(sorted_orientation_images.begin(), sorted_orientation_images.end());
  for (const auto& [name, image_id] : sorted_orientation_images) {
    result->orientation_image_ids.push_back(image_id);
  }
  if (result->orientation_image_ids.empty()) {
    return;
  }

  const Sim3d position_only_tgt_from_src = result->tgt_from_src;
  result->orientation_inlier_mask.assign(result->orientation_image_ids.size(),
                                         0);
  for (const image_t image_id : result->orientation_image_ids) {
    result->orientation_residuals_deg.push_back(
        OrientationResidualDeg(src_reconstruction.Image(image_id),
                               *priors_by_image.at(image_id),
                               position_only_tgt_from_src));
  }
  Sim3d first_refinement = position_only_tgt_from_src;
  if (!SolveJointPosePriorAlignment(src_reconstruction,
                                    priors_by_image,
                                    position_image_ids,
                                    result->orientation_image_ids,
                                    position_fallback_stddev,
                                    orientation_fallback_stddev_rad,
                                    &first_refinement)) {
    return;
  }

  std::vector<image_t> orientation_inliers;
  for (size_t i = 0; i < result->orientation_image_ids.size(); ++i) {
    const image_t image_id = result->orientation_image_ids[i];
    const double error_deg =
        OrientationResidualDeg(src_reconstruction.Image(image_id),
                               *priors_by_image.at(image_id),
                               first_refinement);
    if (error_deg <= orientation_max_error_deg) {
      result->orientation_inlier_mask[i] = 1;
      orientation_inliers.push_back(image_id);
    }
  }
  if (orientation_inliers.empty()) {
    return;
  }

  Sim3d final_refinement = first_refinement;
  if (!SolveJointPosePriorAlignment(src_reconstruction,
                                    priors_by_image,
                                    position_image_ids,
                                    orientation_inliers,
                                    position_fallback_stddev,
                                    orientation_fallback_stddev_rad,
                                    &final_refinement)) {
    std::fill(result->orientation_inlier_mask.begin(),
              result->orientation_inlier_mask.end(),
              0);
    return;
  }

  result->tgt_from_src = final_refinement;
  result->orientation_engaged = true;
  result->orientation_residuals_deg.clear();
  for (const image_t image_id : result->orientation_image_ids) {
    result->orientation_residuals_deg.push_back(
        OrientationResidualDeg(src_reconstruction.Image(image_id),
                               *priors_by_image.at(image_id),
                               final_refinement));
  }
}

bool AlignReconstructionsViaReprojections(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double min_inlier_observations,
    const double max_reproj_error,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GE(min_inlier_observations, 0.0);
  THROW_CHECK_LE(min_inlier_observations, 1.0);

  RANSACOptions ransac_options;
  ransac_options.max_error = 1.0 - min_inlier_observations;
  ransac_options.min_inlier_ratio = 0.2;

  LORANSAC<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator>
      ransac(ransac_options,
             ReconstructionAlignmentEstimator(
                 max_reproj_error, &src_reconstruction, &tgt_reconstruction),
             ReconstructionAlignmentEstimator(
                 max_reproj_error, &src_reconstruction, &tgt_reconstruction));

  const std::vector<std::pair<image_t, image_t>> common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);

  if (common_image_ids.size() < 3) {
    return false;
  }

  std::vector<const Image*> src_images(common_image_ids.size());
  std::vector<const Image*> tgt_images(common_image_ids.size());
  for (size_t i = 0; i < common_image_ids.size(); ++i) {
    src_images[i] = &src_reconstruction.Image(common_image_ids[i].first);
    tgt_images[i] = &tgt_reconstruction.Image(common_image_ids[i].second);
  }

  const auto report = ransac.Estimate(src_images, tgt_images);

  if (report.success) {
    *tgt_from_src = report.model;
  }

  return report.success;
}

bool AlignReconstructionsViaProjCenters(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double max_proj_center_error,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GT(max_proj_center_error, 0);

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_proj_centers;
  for (const auto& image : tgt_reconstruction.Images()) {
    if (image.second.HasPose()) {
      ref_image_names.push_back(image.second.Name());
      ref_proj_centers.push_back(image.second.ProjectionCenter());
    }
  }

  Sim3d tform;
  RANSACOptions ransac_options;
  ransac_options.max_error = max_proj_center_error;
  return AlignReconstructionToLocations(src_reconstruction,
                                        ref_image_names,
                                        ref_proj_centers,
                                        /*min_common_images=*/3,
                                        ransac_options,
                                        tgt_from_src);
}

std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgt_from_src) {
  const std::vector<std::pair<image_t, image_t>> common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);
  const int num_common_images = common_image_ids.size();
  std::vector<ImageAlignmentError> errors;
  errors.reserve(num_common_images);
  for (const auto& image_ids : common_image_ids) {
    const auto& src_image = src_reconstruction.Image(image_ids.first);
    const Rigid3d tgt_world_from_src_cam =
        Inverse(TransformCameraWorld(tgt_from_src, src_image.CamFromWorld()));
    const Rigid3d tgt_world_from_tgt_cam =
        Inverse(tgt_reconstruction.Image(image_ids.second).CamFromWorld());

    ImageAlignmentError error;
    error.image_name = src_image.Name();
    error.rotation_error_deg =
        RadToDeg(tgt_world_from_src_cam.rotation().angularDistance(
            tgt_world_from_tgt_cam.rotation()));
    error.proj_center_error = (tgt_world_from_src_cam.translation() -
                               tgt_world_from_tgt_cam.translation())
                                  .norm();
    errors.push_back(error);
  }
  return errors;
}

bool AlignReconstructionsViaPoints(const Reconstruction& src_reconstruction,
                                   const Reconstruction& tgt_reconstruction,
                                   const size_t min_common_observations,
                                   const double max_error,
                                   const double min_inlier_ratio,
                                   Sim3d* tgt_from_src) {
  THROW_CHECK_GT(min_common_observations, 0);
  THROW_CHECK_GT(max_error, 0.0);
  THROW_CHECK_GE(min_inlier_ratio, 0.0);
  THROW_CHECK_LE(min_inlier_ratio, 1.0);

  std::vector<Eigen::Vector3d> src_xyz;
  std::vector<Eigen::Vector3d> tgt_xyz;
  FlatHashMap<point3D_t, size_t> counts;
  // Associate 3D points using point2D_idx
  for (const auto& src_point3D : src_reconstruction.Points3D()) {
    counts.clear();
    // Count how often a 3D point in tgt is associated to this 3D point.
    for (const auto& track_el : src_point3D.second.track.Elements()) {
      const Image& tgt_image = tgt_reconstruction.Image(track_el.image_id);
      if (!tgt_image.HasPose()) {
        continue;
      }
      const Point2D& tgt_point2D = tgt_image.Point2D(track_el.point2D_idx);
      if (tgt_point2D.HasPoint3D()) {
        if (counts.find(tgt_point2D.point3D_id) != counts.end()) {
          counts[tgt_point2D.point3D_id]++;
        } else {
          counts[tgt_point2D.point3D_id] = 0;
        }
      }
    }
    if (counts.empty()) {
      continue;
    }
    // The 3D point in tgt who is associated the most is selected
    auto best_point3D =
        std::max_element(counts.begin(),
                         counts.end(),
                         [](const std::pair<point3D_t, size_t>& p1,
                            const std::pair<point3D_t, size_t>& p2) {
                           return p1.second < p2.second;
                         });
    if (best_point3D->second >= min_common_observations) {
      src_xyz.push_back(src_point3D.second.xyz);
      tgt_xyz.push_back(tgt_reconstruction.Point3D(best_point3D->first).xyz);
    }
  }
  THROW_CHECK_EQ(src_xyz.size(), tgt_xyz.size());
  LOG(INFO) << "Found " << src_xyz.size() << " / "
            << src_reconstruction.NumPoints3D() << " valid correspondences.";

  RANSACOptions ransac_options;
  ransac_options.max_error = max_error;
  ransac_options.min_inlier_ratio = min_inlier_ratio;
  const auto report =
      EstimateSim3dRobust(src_xyz, tgt_xyz, ransac_options, *tgt_from_src);
  return report.success;
}

namespace {

void CopyRegisteredImage(image_t image_id,
                         const Sim3d& tgt_from_src,
                         const Reconstruction& src_reconstruction,
                         Reconstruction& tgt_reconstruction) {
  const Image& src_image = src_reconstruction.Image(image_id);
  if (!tgt_reconstruction.ExistsCamera(src_image.CameraId())) {
    tgt_reconstruction.AddCamera(
        src_reconstruction.Camera(src_image.CameraId()));
  }
  if (!tgt_reconstruction.ExistsRig(src_image.FramePtr()->RigId())) {
    tgt_reconstruction.AddRig(
        src_reconstruction.Rig(src_image.FramePtr()->RigId()));
  }
  if (!tgt_reconstruction.ExistsFrame(src_image.FrameId())) {
    Frame tgt_frame = src_reconstruction.Frame(src_image.FrameId());
    tgt_frame.ResetRigPtr();
    tgt_reconstruction.AddFrame(std::move(tgt_frame));
    const Rigid3d cam_from_tgt_world =
        TransformCameraWorld(tgt_from_src, src_image.CamFromWorld());
    tgt_reconstruction.Frame(src_image.FrameId())
        .SetCamFromWorld(src_image.CameraId(), cam_from_tgt_world);
  }

  Image tgt_image = src_image;
  tgt_image.ResetCameraPtr();
  tgt_image.ResetFramePtr();
  tgt_reconstruction.AddImage(std::move(tgt_image));
}

}  // namespace

bool MergeReconstructions(const double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction& tgt_reconstruction) {
  Sim3d tgt_from_src;
  if (!AlignReconstructionsViaReprojections(src_reconstruction,
                                            tgt_reconstruction,
                                            /*min_inlier_observations=*/0.3,
                                            max_reproj_error,
                                            &tgt_from_src)) {
    return false;
  }

  // Find common and missing images in the two reconstructions.
  FlatHashSet<image_t> common_image_ids;
  common_image_ids.reserve(src_reconstruction.NumRegImages());
  FlatHashSet<image_t> missing_image_ids;
  missing_image_ids.reserve(src_reconstruction.NumRegImages());
  for (const image_t image_id : src_reconstruction.RegImageIds()) {
    if (tgt_reconstruction.ExistsImage(image_id)) {
      common_image_ids.insert(image_id);
    } else {
      missing_image_ids.insert(image_id);
    }
  }

  // Register the missing images in this src_reconstruction.
  for (const auto image_id : missing_image_ids) {
    CopyRegisteredImage(
        image_id, tgt_from_src, src_reconstruction, tgt_reconstruction);
  }

  // Merge the two point clouds using the following two rules:
  //    - copy points to this src_reconstruction with non-conflicting tracks,
  //      i.e. points that do not have an already triangulated observation
  //      in this src_reconstruction.
  //    - merge tracks that are unambiguous, i.e. only merge points in the two
  //      reconstructions if they have a one-to-one mapping.
  // Note that in both cases no cheirality or reprojection test is performed.

  for (const auto& [_, point3D] : src_reconstruction.Points3D()) {
    Track new_track;
    Track old_track;
    FlatHashSet<point3D_t> old_point3D_ids;
    for (const auto& track_el : point3D.track.Elements()) {
      if (common_image_ids.count(track_el.image_id) > 0) {
        const auto& point2D = tgt_reconstruction.Image(track_el.image_id)
                                  .Point2D(track_el.point2D_idx);
        if (point2D.HasPoint3D()) {
          old_track.AddElement(track_el);
          old_point3D_ids.insert(point2D.point3D_id);
        } else {
          new_track.AddElement(track_el);
        }
      } else if (missing_image_ids.count(track_el.image_id) > 0) {
        tgt_reconstruction.Image(track_el.image_id)
            .ResetPoint3DForPoint2D(track_el.point2D_idx);
        new_track.AddElement(track_el);
      }
    }

    const bool create_new_point = new_track.Length() >= 2;
    const bool merge_new_and_old_point =
        (new_track.Length() + old_track.Length()) >= 2 &&
        old_point3D_ids.size() == 1;
    if (create_new_point || merge_new_and_old_point) {
      const Eigen::Vector3d xyz = tgt_from_src * point3D.xyz;
      const auto point3D_id =
          tgt_reconstruction.AddPoint3D(xyz, new_track, point3D.color);
      if (old_point3D_ids.size() == 1) {
        tgt_reconstruction.MergePoints3D(point3D_id, *old_point3D_ids.begin());
      }
    }
  }

  return true;
}

bool AlignReconstructionToOrigRigScales(
    const NodeHashMap<rig_t, Rig>& orig_rigs, Reconstruction* reconstruction) {
  double scale_sum = 0;
  int scale_count = 0;
  for (const auto& [rig_id, orig_rig] : orig_rigs) {
    double scale_sum_rig = 0;
    int scale_count_rig = 0;
    for (auto& [sensor_id, sensor_from_orig_rig] : orig_rig.NonRefSensors()) {
      if (!sensor_from_orig_rig.has_value()) {
        continue;
      }

      // Here we do not include rigs that are panoramic.
      double sensor_from_orig_rig_norm =
          sensor_from_orig_rig->translation().norm();
      if (sensor_from_orig_rig_norm < 1e-6) {
        continue;
      }
      THROW_CHECK(reconstruction->Rig(rig_id).HasSensorFromRig(sensor_id));
      double scale = reconstruction->Rig(rig_id)
                         .SensorFromRig(sensor_id)
                         .translation()
                         .norm() /
                     sensor_from_orig_rig_norm;
      scale_sum_rig += scale;
      ++scale_count_rig;
    }
    if (scale_count_rig > 0) {
      scale_sum += scale_sum_rig / scale_count_rig;
      ++scale_count;
    }
  }
  if (scale_count == 0) {
    return false;
  }
  Sim3d new_from_old_world;
  new_from_old_world.scale() = scale_count / scale_sum;
  reconstruction->Transform(new_from_old_world);
  return true;
}

AlignmentErrorSummary AlignmentErrorSummary::Compute(
    const std::vector<ImageAlignmentError>& errors) {
  AlignmentErrorSummary summary;
  if (errors.empty()) {
    return summary;
  }

  std::vector<double> rotation_errors_deg;
  rotation_errors_deg.reserve(errors.size());
  std::vector<double> proj_center_errors;
  proj_center_errors.reserve(errors.size());

  for (const auto& error : errors) {
    rotation_errors_deg.push_back(error.rotation_error_deg);
    proj_center_errors.push_back(error.proj_center_error);
  }

  auto ComputeStatistics = [](std::vector<double>& values) {
    Statistics stats;
    if (values.empty()) {
      return stats;
    }
    stats.min = Percentile(values, 0);
    stats.max = Percentile(values, 100);
    stats.mean = Mean(values);
    stats.median = Median(values);
    stats.p90 = Percentile(values, 90);
    stats.p99 = Percentile(values, 99);
    return stats;
  };

  summary.rotation_errors_deg = ComputeStatistics(rotation_errors_deg);
  summary.proj_center_errors = ComputeStatistics(proj_center_errors);
  return summary;
}

}  // namespace colmap
