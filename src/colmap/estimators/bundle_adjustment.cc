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

#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"

namespace colmap {

bool BundleAdjustmentSummary::IsSolutionUsable() const {
  return termination_type == BundleAdjustmentTerminationType::CONVERGENCE ||
         termination_type == BundleAdjustmentTerminationType::NO_CONVERGENCE ||
         termination_type == BundleAdjustmentTerminationType::USER_SUCCESS;
}

std::string BundleAdjustmentSummary::BriefReport() const {
  return "Bundle adjustment report: termination=" +
         std::string(
             BundleAdjustmentTerminationTypeToString(termination_type)) +
         ", num_residuals=" + std::to_string(num_residuals);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

void BundleAdjustmentConfig::FixGauge(BundleAdjustmentGauge gauge) {
  fixed_gauge_ = gauge;
}

BundleAdjustmentGauge BundleAdjustmentConfig::FixedGauge() const {
  return fixed_gauge_;
}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCamIntrinsics() const {
  return constant_cam_intrinsics_.size();
}

size_t BundleAdjustmentConfig::NumConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    const auto& image = reconstruction.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (point2D.HasPoint3D() && !IsIgnoredPoint(point2D.point3D_id)) {
        ++num_observations;
      }
    }
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        ++num_observations_for_point;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  CHECK_GE(num_observations, 0);

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamIntrinsics(
    const camera_t camera_id) {
  constant_cam_intrinsics_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamIntrinsics(
    const camera_t camera_id) {
  constant_cam_intrinsics_.erase(camera_id);
}

bool BundleAdjustmentConfig::HasConstantCamIntrinsics(
    const camera_t camera_id) const {
  return constant_cam_intrinsics_.find(camera_id) !=
         constant_cam_intrinsics_.end();
}

void BundleAdjustmentConfig::SetConstantSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.insert(sensor_id);
}

void BundleAdjustmentConfig::SetVariableSensorFromRigPose(
    const sensor_t sensor_id) {
  constant_sensor_from_rig_poses_.erase(sensor_id);
}

bool BundleAdjustmentConfig::HasConstantSensorFromRigPose(
    const sensor_t sensor_id) const {
  return constant_sensor_from_rig_poses_.find(sensor_id) !=
         constant_sensor_from_rig_poses_.end();
}

void BundleAdjustmentConfig::SetConstantRigFromWorldPose(
    const frame_t frame_id) {
  constant_rig_from_world_poses_.insert(frame_id);
}

void BundleAdjustmentConfig::SetVariableRigFromWorldPose(
    const frame_t frame_id) {
  constant_rig_from_world_poses_.erase(frame_id);
}

bool BundleAdjustmentConfig::HasConstantRigFromWorldPose(
    const frame_t frame_id) const {
  return constant_rig_from_world_poses_.find(frame_id) !=
         constant_rig_from_world_poses_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::unordered_set<camera_t>&
BundleAdjustmentConfig::ConstantCamIntrinsics() const {
  return constant_cam_intrinsics_;
}

const std::unordered_set<sensor_t>&
BundleAdjustmentConfig::ConstantSensorFromRigPoses() const {
  return constant_sensor_from_rig_poses_;
}

const std::unordered_set<frame_t>&
BundleAdjustmentConfig::ConstantRigFromWorldPoses() const {
  return constant_rig_from_world_poses_;
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  THROW_CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::IgnorePoint(const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  CHECK(!HasConstantPoint(point3D_id));
  ignored_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.count(point3D_id);
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.count(point3D_id);
}

bool BundleAdjustmentConfig::IsIgnoredPoint(const point3D_t point3D_id) const {
  return ignored_point3D_ids_.count(point3D_id);
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  THROW_CHECK(options_.Check());
}

const BundleAdjustmentOptions& BundleAdjuster::Options() const {
  return options_;
}

const BundleAdjustmentConfig& BundleAdjuster::Config() const { return config_; }

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentBackendOptions::BundleAdjustmentBackendOptions()
    : ceres(std::make_shared<CeresBundleAdjustmentOptions>()) {}

BundleAdjustmentBackendOptions::BundleAdjustmentBackendOptions(
    const BundleAdjustmentBackendOptions& other) {
  if (other.ceres) {
    ceres = std::make_shared<CeresBundleAdjustmentOptions>(*other.ceres);
  }
}

BundleAdjustmentBackendOptions& BundleAdjustmentBackendOptions::operator=(
    const BundleAdjustmentBackendOptions& other) {
  if (this == &other) {
    return *this;
  }
  if (other.ceres) {
    ceres = std::make_shared<CeresBundleAdjustmentOptions>(*other.ceres);
  } else {
    ceres.reset();
  }
  return *this;
}

bool BundleAdjustmentOptions::Check() const {
  return THROW_CHECK_NOTNULL(ceres)->Check();
}

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  switch (options.backend) {
    case BundleAdjustmentBackend::CERES:
      return CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  }
  LOG(FATAL_THROW) << "Unknown bundle adjustment backend: "
                   << static_cast<int>(options.backend);
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// PosePriorBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

PosePriorBundleAdjustmentBackendOptions::
    PosePriorBundleAdjustmentBackendOptions()
    : ceres(std::make_shared<CeresPosePriorBundleAdjustmentOptions>()) {}

PosePriorBundleAdjustmentBackendOptions::
    PosePriorBundleAdjustmentBackendOptions(
        const PosePriorBundleAdjustmentBackendOptions& other) {
  if (other.ceres) {
    ceres =
        std::make_shared<CeresPosePriorBundleAdjustmentOptions>(*other.ceres);
  }
}

PosePriorBundleAdjustmentBackendOptions&
PosePriorBundleAdjustmentBackendOptions::operator=(
    const PosePriorBundleAdjustmentBackendOptions& other) {
  if (this == &other) {
    return *this;
  }
  if (other.ceres) {
    ceres =
        std::make_shared<CeresPosePriorBundleAdjustmentOptions>(*other.ceres);
  } else {
    ceres.reset();
  }
  return *this;
}

bool PosePriorBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GT(prior_position_fallback_stddev, 0);
  return THROW_CHECK_NOTNULL(ceres)->Check();
}

std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  switch (options.backend) {
    case BundleAdjustmentBackend::CERES:
      return CreatePosePriorCeresBundleAdjuster(options,
                                                prior_options,
                                                config,
                                                std::move(pose_priors),
                                                reconstruction);
  }
  LOG(FATAL_THROW) << "Unknown bundle adjustment backend: "
                   << static_cast<int>(options.backend);
  return nullptr;
}

}  // namespace colmap
