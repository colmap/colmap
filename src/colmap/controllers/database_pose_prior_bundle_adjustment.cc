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

#include "colmap/controllers/database_pose_prior_bundle_adjustment.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/image.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include <Eigen/Cholesky>
#include <ceres/ceres.h>
#include <sqlite3.h>

namespace colmap {
namespace {

struct SqliteDatabaseDeleter {
  void operator()(sqlite3* database) const {
    if (database != nullptr) {
      sqlite3_close_v2(database);
    }
  }
};

struct SqliteStatementDeleter {
  void operator()(sqlite3_stmt* statement) const {
    if (statement != nullptr) {
      sqlite3_finalize(statement);
    }
  }
};

using SqliteDatabasePtr = std::unique_ptr<sqlite3, SqliteDatabaseDeleter>;
using SqliteStatementPtr =
    std::unique_ptr<sqlite3_stmt, SqliteStatementDeleter>;

template <typename MatrixType>
bool ReadStaticMatrixBlob(sqlite3_stmt* statement,
                          const int column,
                          MatrixType* matrix) {
  THROW_CHECK_NOTNULL(matrix);
  if (sqlite3_column_type(statement, column) == SQLITE_NULL) {
    return false;
  }

  const int num_bytes = sqlite3_column_bytes(statement, column);
  const int expected_num_bytes =
      static_cast<int>(matrix->size() * sizeof(typename MatrixType::Scalar));
  if (num_bytes != expected_num_bytes) {
    return false;
  }

  std::memcpy(matrix->data(), sqlite3_column_blob(statement, column), num_bytes);
  return true;
}

std::unordered_set<std::string> ReadPosePriorColumns(sqlite3* database) {
  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(database,
                         "PRAGMA table_info(pose_priors);",
                         -1,
                         &raw_statement,
                         nullptr) != SQLITE_OK) {
    return {};
  }
  SqliteStatementPtr statement(raw_statement);

  std::unordered_set<std::string> columns;
  while (sqlite3_step(statement.get()) == SQLITE_ROW) {
    const auto* text = sqlite3_column_text(statement.get(), 1);
    if (text != nullptr) {
      columns.emplace(reinterpret_cast<const char*>(text));
    }
  }
  return columns;
}

std::optional<std::string> FindColumn(
    const std::unordered_set<std::string>& columns,
    const std::initializer_list<const char*> candidates) {
  for (const char* candidate : candidates) {
    if (columns.count(candidate) > 0) {
      return std::string(candidate);
    }
  }
  return std::nullopt;
}

std::string QuoteIdentifier(const std::string& identifier) {
  return "\"" + identifier + "\"";
}

bool MakeValidCovariance(const double fallback_stddev,
                         Eigen::Matrix3d* covariance) {
  THROW_CHECK_NOTNULL(covariance);
  *covariance = 0.5 * (*covariance + covariance->transpose());
  if (covariance->allFinite()) {
    const Eigen::LLT<Eigen::Matrix3d> llt(*covariance);
    if (llt.info() == Eigen::Success) {
      return true;
    }
  }

  if (!(fallback_stddev > 0.0) || !std::isfinite(fallback_stddev)) {
    return false;
  }
  *covariance = fallback_stddev * fallback_stddev * Eigen::Matrix3d::Identity();
  return true;
}

struct AbsolutePoseRotationPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePoseRotationPriorCostFunctor, 3, 7> {
 public:
  explicit AbsolutePoseRotationPriorCostFunctor(
      const Eigen::Quaterniond& sensor_from_world_rotation_prior)
      : world_from_sensor_rotation_prior_(
            sensor_from_world_rotation_prior.conjugate()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world) *
        world_from_sensor_rotation_prior_.cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);
    return true;
  }

 private:
  const Eigen::Quaterniond world_from_sensor_rotation_prior_;
};

struct AbsoluteRigPoseRotationPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPoseRotationPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
 public:
  explicit AbsoluteRigPoseRotationPriorCostFunctor(
      const Eigen::Quaterniond& sensor_from_world_rotation_prior)
      : world_from_sensor_rotation_prior_(
            sensor_from_world_rotation_prior.conjugate()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Quaternion<T> param_from_prior_rotation =
        sensor_from_world_rotation *
        world_from_sensor_rotation_prior_.cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);
    return true;
  }

 private:
  const Eigen::Quaterniond world_from_sensor_rotation_prior_;
};

class DatabasePosePriorBundleAdjuster : public BundleAdjuster {
 public:
  DatabasePosePriorBundleAdjuster(
      const BundleAdjustmentOptions& options,
      const DatabasePosePriorBundleAdjustmentOptions& prior_options,
      BundleAdjustmentConfig config,
      std::vector<DatabasePosePrior> pose_priors,
      const Sim3d& normalized_from_metric,
      Reconstruction& reconstruction)
      : BundleAdjuster(options, std::move(config)),
        prior_options_(prior_options),
        pose_priors_(std::move(pose_priors)),
        normalized_from_metric_(normalized_from_metric),
        reconstruction_(reconstruction),
        position_loss_function_(
            std::make_unique<ceres::CauchyLoss>(
                prior_options_.prior_position_loss_scale)),
        rotation_loss_function_(
            std::make_unique<ceres::CauchyLoss>(
                prior_options_.prior_rotation_loss_scale)) {
    default_bundle_adjuster_ = CreateDefaultCeresBundleAdjuster(
        options_, config_, reconstruction_);
    AddPosePriorsToProblem();
  }

  ~DatabasePosePriorBundleAdjuster() override { RestoreMetricFrame(); }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    try {
      auto summary = default_bundle_adjuster_->Solve();
      RestoreMetricFrame();
      return summary;
    } catch (...) {
      RestoreMetricFrame();
      throw;
    }
  }

 private:
  void AddPosePriorsToProblem() {
    ceres::Problem& problem = *default_bundle_adjuster_->Problem();
    size_t num_position_priors = 0;
    size_t num_rotation_priors = 0;

    for (const DatabasePosePrior& pose_prior : pose_priors_) {
      Image& image = reconstruction_.Image(pose_prior.image_id);
      Frame& frame = *image.FramePtr();
      Rigid3d& rig_from_world = frame.RigFromWorld();

      const Eigen::Vector3d normalized_position =
          normalized_from_metric_ * pose_prior.position;
      const Eigen::Matrix3d normalized_from_metric_scaled_rotation =
          normalized_from_metric_.scale() *
          normalized_from_metric_.rotation().toRotationMatrix();
      const Eigen::Matrix3d normalized_position_covariance =
          normalized_from_metric_scaled_rotation *
          pose_prior.position_covariance *
          normalized_from_metric_scaled_rotation.transpose();
      const Eigen::Quaterniond normalized_rotation =
          pose_prior.rotation * normalized_from_metric_.rotation().conjugate();

      if (image.IsRefInFrame()) {
        if (!problem.HasParameterBlock(rig_from_world.params.data()) ||
            problem.IsParameterBlockConstant(rig_from_world.params.data())) {
          continue;
        }

        problem.AddResidualBlock(
            CovarianceWeightedCostFunctor<
                AbsolutePosePositionPriorCostFunctor>::Create(
                normalized_position_covariance, normalized_position),
            position_loss_function_.get(),
            rig_from_world.params.data());
        ++num_position_priors;

        problem.AddResidualBlock(
            CovarianceWeightedCostFunctor<
                AbsolutePoseRotationPriorCostFunctor>::Create(
                pose_prior.rotation_covariance, normalized_rotation),
            rotation_loss_function_.get(),
            rig_from_world.params.data());
        ++num_rotation_priors;
      } else {
        Rigid3d& sensor_from_rig = frame.RigPtr()->SensorFromRig(
            image.CameraPtr()->SensorId());
        const bool has_sensor_from_rig =
            problem.HasParameterBlock(sensor_from_rig.params.data());
        const bool has_rig_from_world =
            problem.HasParameterBlock(rig_from_world.params.data());
        if (!has_sensor_from_rig || !has_rig_from_world ||
            (problem.IsParameterBlockConstant(sensor_from_rig.params.data()) &&
             problem.IsParameterBlockConstant(rig_from_world.params.data()))) {
          continue;
        }

        problem.AddResidualBlock(
            CovarianceWeightedCostFunctor<
                AbsoluteRigPosePositionPriorCostFunctor>::Create(
                normalized_position_covariance, normalized_position),
            position_loss_function_.get(),
            sensor_from_rig.params.data(),
            rig_from_world.params.data());
        ++num_position_priors;

        problem.AddResidualBlock(
            CovarianceWeightedCostFunctor<
                AbsoluteRigPoseRotationPriorCostFunctor>::Create(
                pose_prior.rotation_covariance, normalized_rotation),
            rotation_loss_function_.get(),
            sensor_from_rig.params.data(),
            rig_from_world.params.data());
        ++num_rotation_priors;
      }
    }

    LOG(INFO) << "Added " << num_position_priors
              << " position-prior residual blocks and "
              << num_rotation_priors << " rotation-prior residual blocks.";
  }

  void RestoreMetricFrame() {
    if (!metric_frame_restored_) {
      reconstruction_.Transform(Inverse(normalized_from_metric_));
      metric_frame_restored_ = true;
    }
  }

  DatabasePosePriorBundleAdjustmentOptions prior_options_;
  std::vector<DatabasePosePrior> pose_priors_;
  Sim3d normalized_from_metric_;
  Reconstruction& reconstruction_;
  std::unique_ptr<CeresBundleAdjuster> default_bundle_adjuster_;
  std::unique_ptr<ceres::LossFunction> position_loss_function_;
  std::unique_ptr<ceres::LossFunction> rotation_loss_function_;
  bool metric_frame_restored_ = false;
};

}  // namespace

std::vector<DatabasePosePrior> ReadDatabasePosePriors(
    const std::filesystem::path& database_path,
    const DatabasePosePriorBundleAdjustmentOptions& options) {
  sqlite3* raw_database = nullptr;
  const int open_result = sqlite3_open_v2(database_path.string().c_str(),
                                          &raw_database,
                                          SQLITE_OPEN_READONLY,
                                          nullptr);
  SqliteDatabasePtr database(raw_database);
  if (open_result != SQLITE_OK || database == nullptr) {
    LOG(ERROR) << "Cannot open pose-prior database: " << database_path;
    return {};
  }

  const std::unordered_set<std::string> columns =
      ReadPosePriorColumns(database.get());
  if (columns.empty()) {
    LOG(ERROR) << "Database has no pose_priors table: " << database_path;
    return {};
  }

  const auto rotation_column = FindColumn(
      columns,
      {"rotation", "rotation_quaternion", "rotation_prior", "prior_qvec", "qvec"});
  const auto rotation_covariance_column =
      FindColumn(columns,
                 {"rotation_covariance",
                  "rotation_prior_covariance",
                  "prior_qvec_covariance",
                  "qvec_covariance"});
  if (!rotation_column.has_value()) {
    LOG(ERROR) << "pose_priors does not contain a supported quaternion rotation "
                  "column.";
    return {};
  }
  if (!rotation_covariance_column.has_value()) {
    LOG(WARNING)
        << "pose_priors has no supported rotation covariance column; using "
           "the fallback rotation standard deviation.";
  }

  if (columns.count("corr_data_id") == 0 ||
      columns.count("corr_sensor_type") == 0 ||
      columns.count("position") == 0 ||
      columns.count("position_covariance") == 0) {
    LOG(ERROR) << "pose_priors is missing standard COLMAP position-prior "
                  "columns.";
    return {};
  }

  const std::string coordinate_system_expression =
      columns.count("coordinate_system") > 0 ? "coordinate_system" : "-1";
  const std::string rotation_covariance_expression =
      rotation_covariance_column.has_value()
          ? QuoteIdentifier(*rotation_covariance_column)
          : "NULL";
  const std::string query =
      "SELECT corr_data_id, corr_sensor_type, position, "
      "position_covariance, " +
      coordinate_system_expression + ", " +
      QuoteIdentifier(*rotation_column) + ", " +
      rotation_covariance_expression + " FROM pose_priors;";

  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(database.get(),
                         query.c_str(),
                         -1,
                         &raw_statement,
                         nullptr) != SQLITE_OK) {
    LOG(ERROR) << "Failed to prepare pose-prior query: "
               << sqlite3_errmsg(database.get());
    return {};
  }
  SqliteStatementPtr statement(raw_statement);

  std::vector<DatabasePosePrior> pose_priors;
  while (sqlite3_step(statement.get()) == SQLITE_ROW) {
    if (static_cast<SensorType>(sqlite3_column_int(statement.get(), 1)) !=
        SensorType::CAMERA) {
      continue;
    }

    const int coordinate_system = sqlite3_column_int(statement.get(), 4);
    if (coordinate_system ==
        static_cast<int>(PosePrior::CoordinateSystem::WGS84)) {
      LOG_FIRST_N(WARNING, 1)
          << "Ignoring WGS84 pose priors. Convert them to Cartesian "
             "coordinates before bundle adjustment.";
      continue;
    }

    DatabasePosePrior pose_prior;
    pose_prior.image_id =
        static_cast<image_t>(sqlite3_column_int64(statement.get(), 0));
    if (!ReadStaticMatrixBlob(statement.get(), 2, &pose_prior.position)) {
      continue;
    }

    if (!ReadStaticMatrixBlob(
            statement.get(), 3, &pose_prior.position_covariance)) {
      pose_prior.position_covariance = Eigen::Matrix3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
    }
    if (!MakeValidCovariance(options.prior_position_fallback_stddev,
                             &pose_prior.position_covariance)) {
      continue;
    }

    Eigen::Vector4d quaternion_wxyz;
    if (!ReadStaticMatrixBlob(statement.get(), 5, &quaternion_wxyz) ||
        !quaternion_wxyz.allFinite()) {
      continue;
    }
    pose_prior.rotation = Eigen::Quaterniond(quaternion_wxyz(0),
                                             quaternion_wxyz(1),
                                             quaternion_wxyz(2),
                                             quaternion_wxyz(3));
    if (!(pose_prior.rotation.norm() > 0.0) ||
        !std::isfinite(pose_prior.rotation.norm())) {
      continue;
    }
    pose_prior.rotation.normalize();

    if (!ReadStaticMatrixBlob(
            statement.get(), 6, &pose_prior.rotation_covariance)) {
      pose_prior.rotation_covariance = Eigen::Matrix3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
    }
    if (!MakeValidCovariance(options.prior_rotation_fallback_stddev_rad,
                             &pose_prior.rotation_covariance)) {
      continue;
    }

    pose_priors.push_back(std::move(pose_prior));
  }

  LOG(INFO) << "Read " << pose_priors.size()
            << " full pose priors from " << database_path;
  return pose_priors;
}

std::unique_ptr<BundleAdjuster> CreateDatabasePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const DatabasePosePriorBundleAdjustmentOptions& prior_options,
    BundleAdjustmentConfig config,
    const std::filesystem::path& database_path,
    Reconstruction& reconstruction) {
  if (options.backend != BundleAdjustmentBackend::CERES || !options.ceres) {
    LOG(ERROR) << "Database pose-prior bundle adjustment requires the Ceres "
                  "backend.";
    return nullptr;
  }
  if (!(prior_options.prior_position_loss_scale > 0.0) ||
      !(prior_options.prior_rotation_loss_scale > 0.0)) {
    LOG(ERROR) << "Pose-prior robust loss scales must be positive.";
    return nullptr;
  }

  std::vector<DatabasePosePrior> pose_priors =
      ReadDatabasePosePriors(database_path, prior_options);
  pose_priors.erase(
      std::remove_if(pose_priors.begin(),
                     pose_priors.end(),
                     [&config, &reconstruction](const DatabasePosePrior& prior) {
                       return !reconstruction.ExistsImage(prior.image_id) ||
                              !config.HasImage(prior.image_id) ||
                              !reconstruction.Image(prior.image_id).HasPose();
                     }),
      pose_priors.end());

  if (pose_priors.size() < 3) {
    LOG(ERROR) << "At least three registered images with valid position and "
                  "rotation priors are required; found "
               << pose_priors.size() << ".";
    return nullptr;
  }

  std::vector<PosePrior> position_priors;
  position_priors.reserve(pose_priors.size());
  for (const DatabasePosePrior& database_prior : pose_priors) {
    PosePrior position_prior;
    position_prior.corr_data_id =
        reconstruction.Image(database_prior.image_id).DataId();
    position_prior.position = database_prior.position;
    position_prior.position_covariance =
        database_prior.position_covariance;
    position_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    position_priors.push_back(std::move(position_prior));
  }

  RANSACOptions alignment_options;
  std::vector<double> rms_variances;
  rms_variances.reserve(position_priors.size());
  for (const PosePrior& position_prior : position_priors) {
    rms_variances.push_back(position_prior.position_covariance.trace() / 3.0);
  }
  alignment_options.max_error = std::sqrt(
      kChiSquare95ThreeDof * Median(std::move(rms_variances)));

  Sim3d metric_from_original;
  if (!AlignReconstructionToPosePriors(reconstruction,
                                       position_priors,
                                       alignment_options,
                                       &metric_from_original)) {
    LOG(ERROR) << "Failed to align reconstruction to database position priors.";
    return nullptr;
  }
  reconstruction.Transform(metric_from_original);

  // Position priors define translation and scale, while rotation priors define
  // orientation. Do not add the ordinary hard two-camera gauge constraints.
  config.FixGauge(BundleAdjustmentGauge::UNSPECIFIED);
  const Sim3d normalized_from_metric =
      reconstruction.Normalize(/*fixed_scale=*/true);

  try {
    return std::make_unique<DatabasePosePriorBundleAdjuster>(options,
                                                             prior_options,
                                                             std::move(config),
                                                             std::move(pose_priors),
                                                             normalized_from_metric,
                                                             reconstruction);
  } catch (...) {
    reconstruction.Transform(Inverse(normalized_from_metric));
    throw;
  }
}

}  // namespace colmap
