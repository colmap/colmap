#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/sensor/models.h"
#ifdef CASPAR_ENABLED
#include <caspar_model_adapter.h>  // selected by CMake via configure_file
#include <solver.h>
#endif

namespace colmap {
namespace {

#ifdef CASPAR_USE_DOUBLE
typedef double StorageType;
#else
typedef float StorageType;
#endif

template <typename T1, typename T2>
struct PairHash {
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
  }
};

// Holds all factor data for selected camera model
struct ModelData {
  std::vector<StorageType> calib_data;

  // All-variable factors
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> calib_indices;
  std::vector<unsigned int> point_indices;
  std::vector<StorageType> pixels;
  size_t num_factors = 0;

  // Fixed-pose factors
  std::vector<unsigned int> fp_calib_indices;
  std::vector<unsigned int> fp_point_indices;
  std::vector<StorageType> fp_poses;  // packed quat + translation
  std::vector<StorageType> fp_pixels;
  size_t num_fixed_pose = 0;

  // Fixed-point factors
  std::vector<unsigned int> fpt_pose_indices;
  std::vector<unsigned int> fpt_calib_indices;
  std::vector<StorageType> fpt_points;  // packed x,y,z
  std::vector<StorageType> fpt_pixels;
  size_t num_fixed_point = 0;
};

class CasparBundleAdjuster : public BundleAdjuster {
 public:
  CasparBundleAdjuster(BundleAdjustmentOptions options,
                       BundleAdjustmentConfig config,
                       Reconstruction& reconstruction)
      : BundleAdjuster(options, config), reconstruction_(reconstruction) {
    VLOG(1) << "Using Caspar bundle adjuster";

    BuildObservationCounts();
    BuildCameraFrameIndex();
    BuildFactors();
  }

 private:
  // Index (camera_id, frame_id) -> images to avoid O(images) linear search
  void BuildCameraFrameIndex() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();

      if (!CasparModelAdapter::IsSupported(camera.model_id)) continue;

      auto key = std::make_pair(image.CameraId(), image.FrameId());
      camera_frame_to_images_[key].push_back(image_id);
    }
  }

  void BuildObservationCounts() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();

      if (!CasparModelAdapter::IsSupported(camera.model_id)) {
        LOG(WARNING) << "Skipping image " << image_id
                     << " with unsupported camera model: "
                     << camera.ModelName();
        continue;
      }

      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id)) {
          continue;
        }
        point3D_num_observations_[point2D.point3D_id]++;
      }
    }

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
        Image& image = reconstruction_.Image(track_el.image_id);
        Camera& camera = *image.CameraPtr();

        if (CasparModelAdapter::IsSupported(camera.model_id)) {
          point3D_num_observations_[point3D_id]++;
        }
      }
    }
  }

  void BuildFactors() {
    CreateCalibrationNodes();
    CreatePoseNodes();
    CreatePointNodes();
    AddFactorsInOptimalOrder();
    AddExternalFactors();
  }

  void CreateCalibrationNodes() {
    std::vector<camera_t> sorted_camera_ids;
    for (const image_t image_id : config_.Images()) {
      sorted_camera_ids.push_back(reconstruction_.Image(image_id).CameraId());
    }
    std::sort(sorted_camera_ids.begin(), sorted_camera_ids.end());
    sorted_camera_ids.erase(
        std::unique(sorted_camera_ids.begin(), sorted_camera_ids.end()),
        sorted_camera_ids.end());

    for (const camera_t camera_id : sorted_camera_ids) {
      const Camera& camera = reconstruction_.Camera(camera_id);
      if (!CasparModelAdapter::IsSupported(camera.model_id)) continue;
      GetOrCreateCalibration(camera_id, camera);
    }
  }

  void CreatePoseNodes() {
    std::vector<frame_t> sorted_frame_ids;
    for (const image_t image_id : config_.Images()) {
      sorted_frame_ids.push_back(reconstruction_.Image(image_id).FrameId());
    }
    std::sort(sorted_frame_ids.begin(), sorted_frame_ids.end());
    sorted_frame_ids.erase(
        std::unique(sorted_frame_ids.begin(), sorted_frame_ids.end()),
        sorted_frame_ids.end());

    for (const frame_t frame_id : sorted_frame_ids) {
      GetOrCreatePose(frame_id);
    }
  }

  void CreatePointNodes() {
    std::vector<point3D_t> sorted_point_ids;
    for (const auto& [point_id, _] : reconstruction_.Points3D()) {
      if (!config_.IsIgnoredPoint(point_id)) {
        sorted_point_ids.push_back(point_id);
      }
    }
    std::sort(sorted_point_ids.begin(), sorted_point_ids.end());

    for (const point3D_t point_id : sorted_point_ids) {
      GetOrCreatePoint(point_id, reconstruction_.Point3D(point_id));
    }
  }

  void AddFactorsInOptimalOrder() {
    for (size_t calib_idx = 0; calib_idx < num_calibs_; ++calib_idx) {
      const camera_t camera_id = calib_index_to_camera_[calib_idx];
      const Camera& camera = reconstruction_.Camera(camera_id);
      AddFactorsForCalibration(camera_id, camera.model_id);
    }
  }

  void AddFactorsForCalibration(camera_t camera_id, CameraModelId model_id) {
    for (size_t pose_idx = 0; pose_idx < num_poses_; ++pose_idx) {
      const frame_t frame_id = pose_index_to_frame_[pose_idx];

      auto key = std::make_pair(camera_id, frame_id);
      auto it = camera_frame_to_images_.find(key);
      if (it == camera_frame_to_images_.end()) continue;

      const std::vector<image_t>& matching_images = it->second;

      for (size_t point_idx = 0; point_idx < num_points_; ++point_idx) {
        const point3D_t point_id = index_to_point_id_[point_idx];
        const Point3D& point3D = reconstruction_.Point3D(point_id);

        for (const image_t image_id : matching_images) {
          const Image& image = reconstruction_.Image(image_id);

          for (const auto& track_el : point3D.track.Elements()) {
            if (track_el.image_id != image_id) continue;

            const Point2D& point2D = image.Point2D(track_el.point2D_idx);
            AddFactorForObservation(
                image, *image.CameraPtr(), point2D, point3D);
            break;
          }
        }
      }
    }
  }

  void AddExternalFactors() {
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
    GetOrCreatePoint(point3D_id, point3D);

    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) continue;

      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();

      if (!CasparModelAdapter::IsSupported(camera.model_id)) {
        LOG(WARNING) << "Skipping external observation with unsupported "
                        "camera model: "
                     << camera.ModelName();
        continue;
      }

      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      AddFactorForObservation(image, camera, point2D, point3D);
      // Mark camera so AreIntrinsicsVariable() treats it as fixed: we have
      // no pose node for this image so optimizing its intrinsics is undefined.
      cameras_from_outside_config_.insert(camera.camera_id);
    }
  }

  void AddFactorForObservation(const Image& image,
                               const Camera& camera,
                               const Point2D& point2D,
                               const Point3D& point3D) {
    if (!CasparModelAdapter::IsSupported(camera.model_id)) return;

    const bool pose_var = IsPoseVariable(image.FrameId());
    const bool intrinsics_var = AreIntrinsicsVariable(camera.camera_id);
    const bool point_var = IsPointVariable(point2D.point3D_id);

    if (!pose_var && !intrinsics_var && !point_var) return;

    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);

    if (pose_var && intrinsics_var && point_var) {
      // All-variable: pose, calibration, and 3D point all optimized.
      model_data_.pose_indices.push_back(GetOrCreatePose(image.FrameId()));
      model_data_.calib_indices.push_back(calib_idx);
      model_data_.point_indices.push_back(
          GetOrCreatePoint(point2D.point3D_id, point3D));
      model_data_.pixels.push_back(point2D.xy.x());
      model_data_.pixels.push_back(point2D.xy.y());
      ++model_data_.num_factors;

    } else if (pose_var && intrinsics_var && !point_var) {
      // Fixed-point: 3D point is constant, pose and calibration optimized.
      model_data_.fpt_pose_indices.push_back(GetOrCreatePose(image.FrameId()));
      model_data_.fpt_calib_indices.push_back(calib_idx);
      model_data_.fpt_points.push_back(point3D.xyz.x());
      model_data_.fpt_points.push_back(point3D.xyz.y());
      model_data_.fpt_points.push_back(point3D.xyz.z());
      model_data_.fpt_pixels.push_back(point2D.xy.x());
      model_data_.fpt_pixels.push_back(point2D.xy.y());
      ++model_data_.num_fixed_point;

    } else if (!pose_var && intrinsics_var && point_var) {
      // Fixed-pose: pose is constant, calibration and 3D point optimized.
      const Rigid3d& pose = image.FramePtr()->RigFromWorld();
      model_data_.fp_calib_indices.push_back(calib_idx);
      model_data_.fp_point_indices.push_back(
          GetOrCreatePoint(point2D.point3D_id, point3D));
      model_data_.fp_poses.push_back(pose.rotation().x());
      model_data_.fp_poses.push_back(pose.rotation().y());
      model_data_.fp_poses.push_back(pose.rotation().z());
      model_data_.fp_poses.push_back(pose.rotation().w());
      model_data_.fp_poses.push_back(pose.translation().x());
      model_data_.fp_poses.push_back(pose.translation().y());
      model_data_.fp_poses.push_back(pose.translation().z());
      model_data_.fp_pixels.push_back(point2D.xy.x());
      model_data_.fp_pixels.push_back(point2D.xy.y());
      ++model_data_.num_fixed_pose;

    } else {
      LOG(FATAL) << "Unhandled factor combination: pose_var=" << pose_var
                 << " intrinsics_var=" << intrinsics_var
                 << " point_var=" << point_var;
    }
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
      pose_data_.push_back(pose.rotation().x());
      pose_data_.push_back(pose.rotation().y());
      pose_data_.push_back(pose.rotation().z());
      pose_data_.push_back(pose.rotation().w());
      pose_data_.push_back(pose.translation().x());
      pose_data_.push_back(pose.translation().y());
      pose_data_.push_back(pose.translation().z());
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
      // Adapter owns the parameter layout; extracts into model_data_.calib_data
      // in whatever order and stride Caspar expects.
      CasparModelAdapter::ExtractCalib(camera, model_data_.calib_data);
      ++num_calibs_;
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
    // Cameras observed only by images outside the config are treated as fixed:
    // we have no pose nodes for those images, so optimizing their intrinsics
    // would be undefined. Populated by AddFactorsForExternalObservations().
    if (cameras_from_outside_config_.count(camera_id)) return false;
    return true;
  }

  bool IsPointVariable(const point3D_t point3D_id) {
    if (config_.HasConstantPoint(point3D_id)) return false;
    const Point3D point3D = reconstruction_.Point3D(point3D_id);
    size_t num_obs_in_problem = point3D_num_observations_[point3D_id];
    // If the point has track elements outside the current problem, it has
    // external observations and must be treated as fixed (no pose node exists
    // for those images).
    if (point3D.track.Length() > num_obs_in_problem) return false;
    return true;
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    VLOG(2) << "=== CASPAR SOLVER SETUP ===";
    VLOG(2) << "Node counts:";
    VLOG(2) << "  Points: " << num_points_;
    VLOG(2) << "  Poses: " << num_poses_;
    VLOG(2) << "  Calibrations: " << num_calibs_;
    VLOG(2) << "  Factors (all-variable): " << model_data_.num_factors;
    VLOG(2) << "  Factors (fixed-pose): " << model_data_.num_fixed_pose;
    VLOG(2) << "  Factors (fixed-point): " << model_data_.num_fixed_point;

    if (num_points_ > 0)
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    if (num_poses_ > 0)
      solver.set_Pose_nodes_from_stacked_host(pose_data_.data(), 0, num_poses_);
    if (num_calibs_ > 0)
      CasparModelAdapter::SetCalibNodes(
          solver, model_data_.calib_data.data(), num_calibs_);

    // Each Set*FactorIndices call also invokes the corresponding set_*_num()
    // on the solver (handled inside the adapter).
    if (model_data_.num_factors > 0)
      CasparModelAdapter::SetFactorIndices(solver,
                                           model_data_.pose_indices.data(),
                                           model_data_.calib_indices.data(),
                                           model_data_.point_indices.data(),
                                           model_data_.pixels.data(),
                                           model_data_.num_factors);

    if (model_data_.num_fixed_pose > 0)
      CasparModelAdapter::SetFixedPoseFactorIndices(
          solver,
          model_data_.fp_calib_indices.data(),
          model_data_.fp_point_indices.data(),
          model_data_.fp_poses.data(),
          model_data_.fp_pixels.data(),
          model_data_.num_fixed_pose);

    if (model_data_.num_fixed_point > 0)
      CasparModelAdapter::SetFixedPointFactorIndices(
          solver,
          model_data_.fpt_pose_indices.data(),
          model_data_.fpt_calib_indices.data(),
          model_data_.fpt_points.data(),
          model_data_.fpt_pixels.data(),
          model_data_.num_fixed_point);

    solver.finish_indices();
    VLOG(2) << "Solver setup complete";
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0)
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);
    if (num_poses_ > 0)
      solver.get_Pose_nodes_to_stacked_host(pose_data_.data(), 0, num_poses_);
    if (num_calibs_ > 0)
      CasparModelAdapter::GetCalibNodes(
          solver, model_data_.calib_data.data(), num_calibs_);
  }

  void WriteCalibsToReconstruction() {
    for (const auto& [idx, camera_id] : calib_index_to_camera_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;
      Camera& camera = reconstruction_.Camera(camera_id);
      // Adapter owns the layout — WriteCalib knows the stride and param order.
      CasparModelAdapter::WriteCalib(
          camera, model_data_.calib_data.data(), idx);
      THROW_CHECK(camera.VerifyParams());
    }
  }

  void WriteResultsToReconstruction() {
    for (const auto& [idx, point_id] : index_to_point_id_) {
      if (config_.HasConstantPoint(point_id)) continue;
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

    for (const auto& [idx, frame_id] : pose_index_to_frame_) {
      if (!IsPoseVariable(frame_id)) continue;
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      pose.rotation().x() = pose_data_[idx * 7 + 0];
      pose.rotation().y() = pose_data_[idx * 7 + 1];
      pose.rotation().z() = pose_data_[idx * 7 + 2];
      pose.rotation().w() = pose_data_[idx * 7 + 3];
      pose.translation().x() = pose_data_[idx * 7 + 4];
      pose.translation().y() = pose_data_[idx * 7 + 5];
      pose.translation().z() = pose_data_[idx * 7 + 6];
      pose.rotation().normalize();
    }

    WriteCalibsToReconstruction();
  }

  size_t ComputeTotalResiduals() const {
    return 2 * (model_data_.num_factors + model_data_.num_fixed_pose +
                model_data_.num_fixed_point);
  }

  bool ValidateData() {
    if (num_points_ == 0 && num_poses_ == 0 && num_calibs_ == 0) {
      LOG(WARNING) << "No data to optimize";
      return false;
    }
    if (ComputeTotalResiduals() == 0) {
      LOG(WARNING) << "No residuals to optimize";
      return false;
    }
    return true;
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    if (!ValidateData()) {
      auto summary = std::make_shared<BundleAdjustmentSummary>();
      summary->termination_type = BundleAdjustmentTerminationType::USER_FAILURE;
      return summary;
    }

    caspar::SolveResult result;

    // TODO: tordnat implement f64 support

    caspar::SolverParams<StorageType> params;
    auto solver = CreateSolver(params,
                               num_calibs_,
                               num_points_,
                               num_poses_,
                               model_data_.num_factors,
                               model_data_.num_fixed_pose,
                               model_data_.num_fixed_point);
    SetupSolverData(solver);
    result = solver.solve(true);
    ReadSolverResults(solver);

    WriteResultsToReconstruction();

    auto summary = CasparBundleAdjustmentSummary::Create(result);
    summary->num_residuals = ComputeTotalResiduals();
    return summary;
  }

 private:
  ModelData model_data_;
  size_t num_calibs_ = 0;
  std::unordered_map<camera_t, size_t> camera_to_calib_index_;
  std::unordered_map<size_t, camera_t> calib_index_to_camera_;

  Reconstruction& reconstruction_;
  std::unordered_set<camera_t> cameras_from_outside_config_;
  std::unordered_map<std::pair<camera_t, frame_t>,
                     std::vector<image_t>,
                     PairHash<camera_t, frame_t>>
      camera_frame_to_images_;

  size_t num_points_ = 0;
  size_t num_poses_ = 0;
  std::vector<StorageType> point_data_;
  std::vector<StorageType> pose_data_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;
  std::unordered_map<frame_t, size_t> frame_to_pose_index_;
  std::unordered_map<size_t, frame_t> pose_index_to_frame_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
};

}  // namespace

std::shared_ptr<CasparBundleAdjustmentSummary>
CasparBundleAdjustmentSummary::Create(
    const caspar::SolveResult& caspar_summary) {
  auto summary = std::make_shared<CasparBundleAdjustmentSummary>();
  switch (caspar_summary.exit_reason) {
    case (caspar::ExitReason::CONVERGED_DIAG_EXIT):
    case (caspar::ExitReason::CONVERGED_SCORE_THRESHOLD):
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case (caspar::ExitReason::MAX_ITERATIONS):
      summary->termination_type =
          BundleAdjustmentTerminationType::NO_CONVERGENCE;
      break;
    default:
      summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
  }
  return summary;
}

std::unique_ptr<BundleAdjuster> CreateDefaultCasparBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  return std::make_unique<CasparBundleAdjuster>(
      options, config, reconstruction);
}

}  // namespace colmap