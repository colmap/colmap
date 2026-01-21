#include "colmap/estimators/caspar_bundle_adjustment.h"

#include "generated/solver.h"
#include "generated/solver_params.h"

namespace colmap {
namespace {

class CasparBundleAdjuster : public BundleAdjuster {
 public:
  CasparBundleAdjuster(BundleAdjustmentOptions options,
                       BundleAdjustmentConfig config,
                       Reconstruction& reconstruction,
                       caspar::SolverParams params)
      : BundleAdjuster(std::move(options), std::move(config)),
        params_(params),
        reconstruction_(reconstruction) {
    LOG(INFO) << "Using Caspar bundle adjuster";

    // PASS 1: Count all observations
    BuildObservationCounts();

    // PASS 2: Create all factors
    BuildFactors();
  }

  void BuildObservationCounts() {
    // Count observations from images in config
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id)) {
          continue;
        }
        point3D_num_observations_[point2D.point3D_id]++;
      }
    }

    // Count observations from explicit points (images outside config)
    for (const auto point3D_id : config_.VariablePoints()) {
      CountExternalObservations(point3D_id);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      CountExternalObservations(point3D_id);
    }
  }

  void CountExternalObservations(const point3D_t point3D_id) {
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (!config_.HasImage(track_el.image_id)) {
        point3D_num_observations_[point3D_id]++;
      }
    }
  }

  void BuildFactors() {
    // Create factors for observations from images in config
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);

      if (image.CameraPtr()->model_id != CameraModelId::kSimpleRadial) {
        LOG(ERROR) << "ERROR! TRIED TO ADD NON SIMPLE RADIAL CAMERA!";
        continue;
      }

      Camera& camera = *image.CameraPtr();

      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id)) {
          continue;
        }

        const Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
        const bool pose_var = IsPoseVariable(image.FrameId());
        const bool intrinsics_var = AreIntrinsicsVariable(camera.camera_id);
        const bool point_var = IsPointVariable(point2D.point3D_id);

        if (!pose_var && !intrinsics_var && !point_var) {
          continue;  // Nothing to optimize
        }

        if (pose_var && intrinsics_var && point_var) {
          AddSimpleRadialFactor(image, camera, point2D, point3D);
        } else if (pose_var && intrinsics_var && !point_var) {
          AddSimpleRadialFixedPointFactor(image, camera, point2D, point3D);
        } else if (!pose_var && intrinsics_var && point_var) {
          AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
        } else {
          LOG(FATAL) << "Unhandled factor combination: pose_var=" << pose_var
                     << " intrinsics_var=" << intrinsics_var
                     << " point_var=" << point_var;
        }
      }
    }

    // Create factors for observations from images outside config (fixed poses)
    for (const auto point3D_id : config_.VariablePoints()) {
      AddFactorsForExternalObservations(point3D_id);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddFactorsForExternalObservations(point3D_id);
    }
  }

  void AddFactorsForExternalObservations(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));

    Point3D& point3D = reconstruction_.Point3D(point3D_id);

    // Ensure point node exists
    GetOrCreatePoint(point3D_id, point3D);

    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) {
        continue;  // Already handled above
      }

      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      // These observations have fixed poses
      AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
      cameras_from_outside_config_.insert(camera.camera_id);
    }
  }

  void AddSimpleRadialFactor(const Image& image,
                             const Camera& camera,
                             const Point2D& point2D,
                             const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t pose_idx = GetOrCreatePose(image.FrameId());
    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);

    simple_radial_pose_indices_.push_back(pose_idx);
    simple_radial_calib_indices_.push_back(calib_idx);
    simple_radial_point_indices_.push_back(point_idx);
    simple_radial_pixels_.push_back(point2D.xy.x());
    simple_radial_pixels_.push_back(point2D.xy.y());
    num_simple_radial_++;
  }

  void AddSimpleRadialFixedPoseFactor(const Image& image,
                                      const Camera& camera,
                                      const Point2D& point2D,
                                      const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);

    simple_radial_fixed_pose_calib_indices_.push_back(calib_idx);
    simple_radial_fixed_pose_point_indices_.push_back(point_idx);

    const Rigid3d& pose = image.FramePtr()->RigFromWorld();
    simple_radial_fixed_pose_poses_.push_back(pose.rotation.x());
    simple_radial_fixed_pose_poses_.push_back(pose.rotation.y());
    simple_radial_fixed_pose_poses_.push_back(pose.rotation.z());
    simple_radial_fixed_pose_poses_.push_back(pose.rotation.w());
    simple_radial_fixed_pose_poses_.push_back(pose.translation.x());
    simple_radial_fixed_pose_poses_.push_back(pose.translation.y());
    simple_radial_fixed_pose_poses_.push_back(pose.translation.z());

    simple_radial_fixed_pose_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_pose_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_pose_++;
  }

  void AddSimpleRadialFixedPointFactor(const Image& image,
                                       const Camera& camera,
                                       const Point2D& point2D,
                                       const Point3D& point3D) {
    const size_t pose_idx = GetOrCreatePose(image.FrameId());
    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);

    simple_radial_fixed_point_pose_indices_.push_back(pose_idx);
    simple_radial_fixed_point_calib_indices_.push_back(calib_idx);
    simple_radial_fixed_point_points_.push_back(point3D.xyz.x());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.y());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.z());

    simple_radial_fixed_point_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_point_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_point_++;
  }

  size_t GetOrCreatePoint(const point3D_t point_id, const Point3D& point) {
    auto [it, inserted] = point_id_to_index_.try_emplace(point_id, num_points_);
    if (inserted) {
      index_to_point_id_[num_points_] = point_id;
      point_data_.push_back(point.xyz.x());
      point_data_.push_back(point.xyz.y());
      point_data_.push_back(point.xyz.z());
      num_points_++;
    }
    return it->second;
  }

  size_t GetOrCreatePose(const frame_t frame_id) {
    auto [it, inserted] =
        frame_to_pose_index_.try_emplace(frame_id, num_poses_);
    if (inserted) {
      pose_index_to_frame_[num_poses_] = frame_id;
      const Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();

      pose_data_.push_back(pose.rotation.x());
      pose_data_.push_back(pose.rotation.y());
      pose_data_.push_back(pose.rotation.z());
      pose_data_.push_back(pose.rotation.w());
      pose_data_.push_back(pose.translation.x());
      pose_data_.push_back(pose.translation.y());
      pose_data_.push_back(pose.translation.z());
      num_poses_++;
    }
    return it->second;
  }

  size_t GetOrCreateCalibration(const camera_t camera_id,
                                const Camera& camera) {
    auto [it, inserted] =
        camera_to_calib_index_.try_emplace(camera_id, num_calibs_);
    if (inserted) {
      calib_index_to_camera_[num_calibs_] = camera_id;

      for (const auto& param : camera.params) {
        calib_data_.push_back(param);
      }
      num_calibs_++;
    }
    return it->second;
  }

  bool IsPoseVariable(const frame_t frame_id) {
    if (!options_.refine_rig_from_world) return false;
    if (config_.HasConstantRigFromWorldPose(frame_id)) return false;
    return true;
  }

  bool AreIntrinsicsVariable(const camera_t camera_id) {
    bool any_refinement = options_.refine_focal_length ||
                          options_.refine_principal_point ||
                          options_.refine_extra_params;
    if (!any_refinement) return false;
    if (config_.HasConstantCamIntrinsics(camera_id)) return false;
    if (cameras_from_outside_config_.count(camera_id)) return false;
    return true;
  }

  bool IsPointVariable(const point3D_t point3D_id) {
    if (config_.HasConstantPoint(point3D_id)) return false;
    const Point3D point3D = reconstruction_.Point3D(point3D_id);
    size_t num_obs_in_problem = point3D_num_observations_[point3D_id];
    if (point3D.track.Length() > num_obs_in_problem) return false;
    return true;
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    LOG(INFO) << "=== CASPAR SOLVER SETUP ===";
    LOG(INFO) << "Node counts:";
    LOG(INFO) << "  Points: " << num_points_;
    LOG(INFO) << "  Poses: " << num_poses_;
    LOG(INFO) << "  Calibrations: " << num_calibs_;

    LOG(INFO) << "Factor counts:";
    LOG(INFO) << "  simple_radial: " << num_simple_radial_;
    LOG(INFO) << "  simple_radial_fixed_pose: "
              << num_simple_radial_fixed_pose_;
    LOG(INFO) << "  simple_radial_fixed_point: "
              << num_simple_radial_fixed_point_;

    size_t total_residuals = ComputeTotalResiduals();
    LOG(INFO) << "Total residuals: " << total_residuals;

    // Set node data
    if (num_points_ > 0) {
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.set_Pose_nodes_from_stacked_host(pose_data_.data(), 0, num_poses_);
    }
    if (num_calibs_ > 0) {
      solver.set_SimpleRadialCalib_nodes_from_stacked_host(
          calib_data_.data(), 0, num_calibs_);
    }

    // Set factor data for simple_radial
    if (num_simple_radial_ > 0) {
      solver.set_simple_radial_pose_indices_from_host(
          simple_radial_pose_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_calib_indices_from_host(
          simple_radial_calib_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_point_indices_from_host(
          simple_radial_point_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_pixel_data_from_stacked_host(
          simple_radial_pixels_.data(), 0, num_simple_radial_);
    }

    // Set factor data for simple_radial_fixed_pose
    if (num_simple_radial_fixed_pose_ > 0) {
      solver.set_simple_radial_fixed_pose_calib_indices_from_host(
          simple_radial_fixed_pose_calib_indices_.data(),
          num_simple_radial_fixed_pose_);
      solver.set_simple_radial_fixed_pose_point_indices_from_host(
          simple_radial_fixed_pose_point_indices_.data(),
          num_simple_radial_fixed_pose_);
      solver.set_simple_radial_fixed_pose_cam_T_world_data_from_stacked_host(
          simple_radial_fixed_pose_poses_.data(),
          0,
          num_simple_radial_fixed_pose_);
      solver.set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
          simple_radial_fixed_pose_pixels_.data(),
          0,
          num_simple_radial_fixed_pose_);
    }

    // Set factor data for simple_radial_fixed_point
    if (num_simple_radial_fixed_point_ > 0) {
      solver.set_simple_radial_fixed_point_pose_indices_from_host(
          simple_radial_fixed_point_pose_indices_.data(),
          num_simple_radial_fixed_point_);
      solver.set_simple_radial_fixed_point_calib_indices_from_host(
          simple_radial_fixed_point_calib_indices_.data(),
          num_simple_radial_fixed_point_);
      solver.set_simple_radial_fixed_point_point_data_from_stacked_host(
          simple_radial_fixed_point_points_.data(),
          0,
          num_simple_radial_fixed_point_);
      solver.set_simple_radial_fixed_point_pixel_data_from_stacked_host(
          simple_radial_fixed_point_pixels_.data(),
          0,
          num_simple_radial_fixed_point_);
    }

    // Set factor counts
    solver.set_simple_radial_num(num_simple_radial_);
    solver.set_simple_radial_fixed_pose_num(num_simple_radial_fixed_pose_);
    solver.set_simple_radial_fixed_point_num(num_simple_radial_fixed_point_);

    solver.finish_indices();
    LOG(INFO) << "Solver setup complete";
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0) {
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.get_Pose_nodes_to_stacked_host(pose_data_.data(), 0, num_poses_);
    }
    if (num_calibs_ > 0) {
      solver.get_SimpleRadialCalib_nodes_to_stacked_host(
          calib_data_.data(), 0, num_calibs_);
    }
  }

  void WriteResultsToReconstruction() {
    // Write back points
    for (const auto& [idx, point_id] : index_to_point_id_) {
      if (config_.HasConstantPoint(point_id)) {
        continue;  // Only skip truly constant points
      }
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

    // Write back poses
    for (const auto& [idx, frame_id] : pose_index_to_frame_) {
      if (!IsPoseVariable(frame_id)) continue;

      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      pose.rotation.x() = pose_data_[idx * 7 + 0];
      pose.rotation.y() = pose_data_[idx * 7 + 1];
      pose.rotation.z() = pose_data_[idx * 7 + 2];
      pose.rotation.w() = pose_data_[idx * 7 + 3];
      pose.translation.x() = pose_data_[idx * 7 + 4];
      pose.translation.y() = pose_data_[idx * 7 + 5];
      pose.translation.z() = pose_data_[idx * 7 + 6];
      pose.rotation.normalize();
    }

    // Write back calibrations (shared across frames with same camera)
    for (const auto& [idx, camera_id] : calib_index_to_camera_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;

      Camera& camera = reconstruction_.Camera(camera_id);
      for (size_t i = 0; i < camera.params.size(); i++) {
        camera.params[i] = calib_data_[idx * 4 + i];
      }
      THROW_CHECK(camera.VerifyParams());
    }
  }

  size_t ComputeTotalResiduals() const {
    return 2 * (num_simple_radial_ + num_simple_radial_fixed_pose_ +
                num_simple_radial_fixed_point_);
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return dummy_problem_; }

  bool ValidateData() {
    if (num_points_ == 0 && num_poses_ == 0 && num_calibs_ == 0) {
      LOG(WARNING) << "No data to optimize";
      return false;
    }

    size_t total_residuals = ComputeTotalResiduals();
    if (total_residuals == 0) {
      LOG(WARNING) << "No residuals to optimize";
      return false;
    }

    return true;
  }

  ceres::Solver::Summary Solve() override {
    if (!ValidateData()) {
      ceres::Solver::Summary summary;
      summary.termination_type = ceres::CONVERGENCE;
      summary.message = "Invalid data for optimization";
      return summary;
    }

    // Debug
    // params_.solver_iter_max = 200;
    // params_.pcg_iter_max = 50;
    caspar::GraphSolver solver =
        caspar::GraphSolver(params_,
                            num_points_,
                            num_poses_,
                            num_calibs_,
                            num_simple_radial_,
                            num_simple_radial_fixed_pose_,
                            num_simple_radial_fixed_point_);

    SetupSolverData(solver);

    LOG(INFO) << "Starting Caspar solver...";
    const float result = solver.solve(false);
    LOG(INFO) << "Solve completed with cost: " << result;

    ReadSolverResults(solver);
    WriteResultsToReconstruction();

    ceres::Solver::Summary summary;
    summary.final_cost = result;
    summary.num_residuals_reduced = ComputeTotalResiduals();
    summary.termination_type = ceres::CONVERGENCE;
    return summary;
  }

 private:
  caspar::SolverParams params_;
  Reconstruction& reconstruction_;
  std::shared_ptr<ceres::Problem> dummy_problem_;

  std::unordered_set<camera_t> cameras_from_outside_config_;

  size_t num_points_ = 0;
  size_t num_poses_ = 0;
  size_t num_calibs_ = 0;

  size_t num_simple_radial_ = 0;
  size_t num_simple_radial_fixed_pose_ = 0;
  size_t num_simple_radial_fixed_point_ = 0;

  std::vector<float> point_data_;
  std::vector<float> pose_data_;
  std::vector<float> calib_data_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;

  std::unordered_map<frame_t, size_t> frame_to_pose_index_;
  std::unordered_map<size_t, frame_t> pose_index_to_frame_;

  std::unordered_map<camera_t, size_t> camera_to_calib_index_;
  std::unordered_map<size_t, camera_t> calib_index_to_camera_;

  std::unordered_map<point3D_t, size_t> point3D_num_observations_;

  std::vector<unsigned int> simple_radial_pose_indices_;
  std::vector<unsigned int> simple_radial_calib_indices_;
  std::vector<unsigned int> simple_radial_point_indices_;
  std::vector<float> simple_radial_pixels_;

  std::vector<unsigned int> simple_radial_fixed_pose_calib_indices_;
  std::vector<unsigned int> simple_radial_fixed_pose_point_indices_;
  std::vector<float> simple_radial_fixed_pose_poses_;
  std::vector<float> simple_radial_fixed_pose_pixels_;

  std::vector<unsigned int> simple_radial_fixed_point_pose_indices_;
  std::vector<unsigned int> simple_radial_fixed_point_calib_indices_;
  std::vector<float> simple_radial_fixed_point_points_;
  std::vector<float> simple_radial_fixed_point_pixels_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreateCasparBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction,
    caspar::SolverParams params = caspar::SolverParams()) {
  return std::make_unique<CasparBundleAdjuster>(
      std::move(options), std::move(config), reconstruction, params);
}

}  // namespace colmap