#include "colmap/estimators/caspar_bundle_adjustment.h"

#include "generated/solver.h"
#include "generated/solver_params.h"

namespace colmap {
namespace {

template <typename T1, typename T2>
struct PairHash {
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
  }
};

class CasparBundleAdjuster : public BundleAdjuster {
 public:
  CasparBundleAdjuster(BundleAdjustmentOptions options,
                       BundleAdjustmentConfig config,
                       Reconstruction& reconstruction,
                       caspar::SolverParams params)
      : BundleAdjuster(std::move(options), std::move(config)),
        params_(params),
        reconstruction_(reconstruction) {
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

      if (camera.model_id != CameraModelId::kSimpleRadial &&
          camera.model_id != CameraModelId::kPinhole) {
        continue;
      }

      auto key = std::make_pair(image.CameraId(), image.FrameId());
      camera_frame_to_images_[key].push_back(image_id);
    }
  }

  void BuildObservationCounts() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();

      if (camera.model_id != CameraModelId::kSimpleRadial &&
          camera.model_id != CameraModelId::kPinhole) {
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

        if (camera.model_id == CameraModelId::kSimpleRadial ||
            camera.model_id == CameraModelId::kPinhole) {
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
      if (camera.model_id == CameraModelId::kSimpleRadial) {
        GetOrCreateSimpleRadialCalibration(camera_id, camera);
      } else if (camera.model_id == CameraModelId::kPinhole) {
        GetOrCreatePinholeCalibration(camera_id, camera);
      }
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
    for (size_t calib_idx = 0; calib_idx < num_simple_radial_calibs_;
         ++calib_idx) {
      const camera_t camera_id =
          simple_radial_calib_index_to_camera_[calib_idx];
      AddFactorsForCalibration(camera_id, CameraModelId::kSimpleRadial);
    }

    for (size_t calib_idx = 0; calib_idx < num_pinhole_calibs_; ++calib_idx) {
      const camera_t camera_id = pinhole_calib_index_to_camera_[calib_idx];
      AddFactorsForCalibration(camera_id, CameraModelId::kPinhole);
    }
  }

  void AddFactorsForCalibration(camera_t camera_id, CameraModelId model_id) {
    const Camera& camera = reconstruction_.Camera(camera_id);

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
            AddFactorForObservation(image, camera, point2D, point3D, model_id);
            break;
          }
        }
      }
    }
  }

  void AddFactorForObservation(const Image& image,
                               const Camera& camera,
                               const Point2D& point2D,
                               const Point3D& point3D,
                               CameraModelId model_id) {
    const bool pose_var = IsPoseVariable(image.FrameId());
    const bool intrinsics_var = AreIntrinsicsVariable(camera.camera_id);
    const bool point_var = IsPointVariable(point2D.point3D_id);

    if (!pose_var && !intrinsics_var && !point_var) return;

    if (model_id == CameraModelId::kSimpleRadial) {
      if (pose_var && intrinsics_var && point_var) {
        AddSimpleRadialFactor(image, camera, point2D, point3D);
      } else if (pose_var && intrinsics_var && !point_var) {
        AddSimpleRadialFixedPointFactor(image, camera, point2D, point3D);
      } else if (!pose_var && intrinsics_var && point_var) {
        AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
      } else {
        LOG(FATAL) << "Unhandled factor combination";
      }
    } else if (model_id == CameraModelId::kPinhole) {
      if (pose_var && intrinsics_var && point_var) {
        AddPinholeFactor(image, camera, point2D, point3D);
      } else if (pose_var && intrinsics_var && !point_var) {
        AddPinholeFixedPointFactor(image, camera, point2D, point3D);
      } else if (!pose_var && intrinsics_var && point_var) {
        AddPinholeFixedPoseFactor(image, camera, point2D, point3D);
      } else {
        LOG(FATAL) << "Unhandled factor combination";
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
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      if (camera.model_id == CameraModelId::kSimpleRadial) {
        AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
        cameras_from_outside_config_.insert(camera.camera_id);
      } else if (camera.model_id == CameraModelId::kPinhole) {
        AddPinholeFixedPoseFactor(image, camera, point2D, point3D);
        cameras_from_outside_config_.insert(camera.camera_id);
      } else {
        LOG(WARNING)
            << "Skipping external observation with unsupported camera model:"
            << camera.ModelName();
      }
    }
  }

  void AddSimpleRadialFactor(const Image& image,
                             const Camera& camera,
                             const Point2D& point2D,
                             const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t pose_idx = GetOrCreatePose(image.FrameId());
    const size_t calib_idx =
        GetOrCreateSimpleRadialCalibration(camera.camera_id, camera);

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
    const size_t calib_idx =
        GetOrCreateSimpleRadialCalibration(camera.camera_id, camera);

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
    const size_t calib_idx =
        GetOrCreateSimpleRadialCalibration(camera.camera_id, camera);

    simple_radial_fixed_point_pose_indices_.push_back(pose_idx);
    simple_radial_fixed_point_calib_indices_.push_back(calib_idx);
    simple_radial_fixed_point_points_.push_back(point3D.xyz.x());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.y());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.z());

    simple_radial_fixed_point_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_point_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_point_++;
  }

  void AddPinholeFactor(const Image& image,
                        const Camera& camera,
                        const Point2D& point2D,
                        const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t pose_idx = GetOrCreatePose(image.FrameId());
    const size_t calib_idx =
        GetOrCreatePinholeCalibration(camera.camera_id, camera);

    pinhole_pose_indices_.push_back(pose_idx);
    pinhole_calib_indices_.push_back(calib_idx);
    pinhole_point_indices_.push_back(point_idx);
    pinhole_pixels_.push_back(point2D.xy.x());
    pinhole_pixels_.push_back(point2D.xy.y());
    num_pinhole_++;
  }

  void AddPinholeFixedPoseFactor(const Image& image,
                                 const Camera& camera,
                                 const Point2D& point2D,
                                 const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t calib_idx =
        GetOrCreatePinholeCalibration(camera.camera_id, camera);

    pinhole_fixed_pose_calib_indices_.push_back(calib_idx);
    pinhole_fixed_pose_point_indices_.push_back(point_idx);

    const Rigid3d& pose = image.FramePtr()->RigFromWorld();
    pinhole_fixed_pose_poses_.push_back(pose.rotation.x());
    pinhole_fixed_pose_poses_.push_back(pose.rotation.y());
    pinhole_fixed_pose_poses_.push_back(pose.rotation.z());
    pinhole_fixed_pose_poses_.push_back(pose.rotation.w());
    pinhole_fixed_pose_poses_.push_back(pose.translation.x());
    pinhole_fixed_pose_poses_.push_back(pose.translation.y());
    pinhole_fixed_pose_poses_.push_back(pose.translation.z());

    pinhole_fixed_pose_pixels_.push_back(point2D.xy.x());
    pinhole_fixed_pose_pixels_.push_back(point2D.xy.y());
    num_pinhole_fixed_pose_++;
  }

  void AddPinholeFixedPointFactor(const Image& image,
                                  const Camera& camera,
                                  const Point2D& point2D,
                                  const Point3D& point3D) {
    const size_t pose_idx = GetOrCreatePose(image.FrameId());
    const size_t calib_idx =
        GetOrCreatePinholeCalibration(camera.camera_id, camera);

    pinhole_fixed_point_pose_indices_.push_back(pose_idx);
    pinhole_fixed_point_calib_indices_.push_back(calib_idx);
    pinhole_fixed_point_points_.push_back(point3D.xyz.x());
    pinhole_fixed_point_points_.push_back(point3D.xyz.y());
    pinhole_fixed_point_points_.push_back(point3D.xyz.z());

    pinhole_fixed_point_pixels_.push_back(point2D.xy.x());
    pinhole_fixed_point_pixels_.push_back(point2D.xy.y());
    num_pinhole_fixed_point_++;
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

  size_t GetOrCreateSimpleRadialCalibration(const camera_t camera_id,
                                            const Camera& camera) {
    auto [it, inserted] = camera_to_simple_radial_calib_index_.try_emplace(
        camera_id, num_simple_radial_calibs_);
    if (inserted) {
      simple_radial_calib_index_to_camera_[num_simple_radial_calibs_] =
          camera_id;

      for (const auto& param : camera.params) {
        simple_radial_calib_data_.push_back(param);
      }
      num_simple_radial_calibs_++;
    }
    return it->second;
  }

  size_t GetOrCreatePinholeCalibration(const camera_t camera_id,
                                       const Camera& camera) {
    auto [it, inserted] = camera_to_pinhole_calib_index_.try_emplace(
        camera_id, num_pinhole_calibs_);
    if (inserted) {
      pinhole_calib_index_to_camera_[num_pinhole_calibs_] = camera_id;

      pinhole_calib_data_.push_back(camera.FocalLengthX());
      pinhole_calib_data_.push_back(camera.FocalLengthY());
      pinhole_calib_data_.push_back(camera.PrincipalPointX());
      pinhole_calib_data_.push_back(camera.PrincipalPointY());

      num_pinhole_calibs_++;
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
    VLOG(2) << "=== CASPAR SOLVER SETUP ===";
    VLOG(2) << "Node counts:";
    VLOG(2) << "  Points: " << num_points_;
    VLOG(2) << "  Poses: " << num_poses_;
    VLOG(2) << "  SimpleRadial Calibrations: " << num_simple_radial_calibs_;
    VLOG(2) << "  Pinhole Calibrations: " << num_pinhole_calibs_;

    VLOG(2) << "Factor counts:";
    VLOG(2) << "  simple_radial: " << num_simple_radial_;
    VLOG(2) << "  simple_radial_fixed_pose: " << num_simple_radial_fixed_pose_;
    VLOG(2) << "  simple_radial_fixed_point: "
            << num_simple_radial_fixed_point_;
    VLOG(2) << "  pinhole: " << num_pinhole_;
    VLOG(2) << "  pinhole_fixed_pose: " << num_pinhole_fixed_pose_;
    VLOG(2) << "  pinhole_fixed_point: " << num_pinhole_fixed_point_;

    size_t total_residuals = ComputeTotalResiduals();
    VLOG(2) << "Total residuals: " << total_residuals;

    if (num_points_ > 0) {
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.set_Pose_nodes_from_stacked_host(pose_data_.data(), 0, num_poses_);
    }
    if (num_simple_radial_calibs_ > 0) {
      solver.set_SimpleRadialCalib_nodes_from_stacked_host(
          simple_radial_calib_data_.data(), 0, num_simple_radial_calibs_);
    }
    if (num_pinhole_calibs_ > 0) {
      solver.set_PinholeCalib_nodes_from_stacked_host(
          pinhole_calib_data_.data(), 0, num_pinhole_calibs_);
    }

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

    if (num_pinhole_ > 0) {
      solver.set_pinhole_pose_indices_from_host(pinhole_pose_indices_.data(),
                                                num_pinhole_);
      solver.set_pinhole_calib_indices_from_host(pinhole_calib_indices_.data(),
                                                 num_pinhole_);
      solver.set_pinhole_point_indices_from_host(pinhole_point_indices_.data(),
                                                 num_pinhole_);
      solver.set_pinhole_pixel_data_from_stacked_host(
          pinhole_pixels_.data(), 0, num_pinhole_);
    }

    if (num_pinhole_fixed_pose_ > 0) {
      solver.set_pinhole_fixed_pose_calib_indices_from_host(
          pinhole_fixed_pose_calib_indices_.data(), num_pinhole_fixed_pose_);
      solver.set_pinhole_fixed_pose_point_indices_from_host(
          pinhole_fixed_pose_point_indices_.data(), num_pinhole_fixed_pose_);
      solver.set_pinhole_fixed_pose_cam_T_world_data_from_stacked_host(
          pinhole_fixed_pose_poses_.data(), 0, num_pinhole_fixed_pose_);
      solver.set_pinhole_fixed_pose_pixel_data_from_stacked_host(
          pinhole_fixed_pose_pixels_.data(), 0, num_pinhole_fixed_pose_);
    }

    if (num_pinhole_fixed_point_ > 0) {
      solver.set_pinhole_fixed_point_pose_indices_from_host(
          pinhole_fixed_point_pose_indices_.data(), num_pinhole_fixed_point_);
      solver.set_pinhole_fixed_point_calib_indices_from_host(
          pinhole_fixed_point_calib_indices_.data(), num_pinhole_fixed_point_);
      solver.set_pinhole_fixed_point_point_data_from_stacked_host(
          pinhole_fixed_point_points_.data(), 0, num_pinhole_fixed_point_);
      solver.set_pinhole_fixed_point_pixel_data_from_stacked_host(
          pinhole_fixed_point_pixels_.data(), 0, num_pinhole_fixed_point_);
    }

    solver.set_simple_radial_num(num_simple_radial_);
    solver.set_simple_radial_fixed_pose_num(num_simple_radial_fixed_pose_);
    solver.set_simple_radial_fixed_point_num(num_simple_radial_fixed_point_);
    solver.set_pinhole_num(num_pinhole_);
    solver.set_pinhole_fixed_pose_num(num_pinhole_fixed_pose_);
    solver.set_pinhole_fixed_point_num(num_pinhole_fixed_point_);

    solver.finish_indices();
    VLOG(2) << "Solver setup complete";
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0) {
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.get_Pose_nodes_to_stacked_host(pose_data_.data(), 0, num_poses_);
    }
    if (num_simple_radial_calibs_ > 0) {
      solver.get_SimpleRadialCalib_nodes_to_stacked_host(
          simple_radial_calib_data_.data(), 0, num_simple_radial_calibs_);
    }
    if (num_pinhole_calibs_ > 0) {
      solver.get_PinholeCalib_nodes_to_stacked_host(
          pinhole_calib_data_.data(), 0, num_pinhole_calibs_);
    }
  }

  void WriteResultsToReconstruction() {
    for (const auto& [idx, point_id] : index_to_point_id_) {
      if (config_.HasConstantPoint(point_id)) {
        continue;
      }
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

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

    for (const auto& [idx, camera_id] : simple_radial_calib_index_to_camera_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;

      Camera& camera = reconstruction_.Camera(camera_id);
      for (size_t i = 0; i < camera.params.size(); i++) {
        camera.params[i] = simple_radial_calib_data_[idx * 4 + i];
      }
      THROW_CHECK(camera.VerifyParams());
    }

    for (const auto& [idx, camera_id] : pinhole_calib_index_to_camera_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;

      Camera& camera = reconstruction_.Camera(camera_id);
      camera.SetFocalLengthX(pinhole_calib_data_[idx * 4 + 0]);
      camera.SetFocalLengthY(pinhole_calib_data_[idx * 4 + 1]);
      camera.SetPrincipalPointX(pinhole_calib_data_[idx * 4 + 2]);
      camera.SetPrincipalPointY(pinhole_calib_data_[idx * 4 + 3]);

      THROW_CHECK(camera.VerifyParams());
    }
  }

  size_t ComputeTotalResiduals() const {
    return 2 * (num_simple_radial_ + num_simple_radial_fixed_pose_ +
                num_simple_radial_fixed_point_ + num_pinhole_ +
                num_pinhole_fixed_pose_ + num_pinhole_fixed_point_);
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return dummy_problem_; }

  bool ValidateData() {
    if (num_points_ == 0 && num_poses_ == 0 && num_simple_radial_calibs_ == 0 &&
        num_pinhole_calibs_ == 0) {
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

    caspar::GraphSolver solver =
        caspar::GraphSolver(params_,
                            num_pinhole_calibs_,
                            num_points_,
                            num_poses_,
                            num_simple_radial_calibs_,
                            num_simple_radial_,
                            num_simple_radial_fixed_pose_,
                            num_simple_radial_fixed_point_,
                            num_pinhole_,
                            num_pinhole_fixed_pose_,
                            num_pinhole_fixed_point_);

    SetupSolverData(solver);

    VLOG(2) << "Starting Caspar solver...";
    const float result = solver.solve(false);
    VLOG(2) << "Solve completed with cost: " << result;

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

  // Lightweight index: (camera_id, frame_id) -> images
  std::unordered_map<std::pair<camera_t, frame_t>,
                     std::vector<image_t>,
                     PairHash<camera_t, frame_t>>
      camera_frame_to_images_;

  size_t num_points_ = 0;
  size_t num_poses_ = 0;
  size_t num_simple_radial_calibs_ = 0;
  size_t num_pinhole_calibs_ = 0;

  size_t num_simple_radial_ = 0;
  size_t num_simple_radial_fixed_pose_ = 0;
  size_t num_simple_radial_fixed_point_ = 0;
  size_t num_pinhole_ = 0;
  size_t num_pinhole_fixed_pose_ = 0;
  size_t num_pinhole_fixed_point_ = 0;

  std::vector<float> point_data_;
  std::vector<float> pose_data_;
  std::vector<float> simple_radial_calib_data_;
  std::vector<float> pinhole_calib_data_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;

  std::unordered_map<frame_t, size_t> frame_to_pose_index_;
  std::unordered_map<size_t, frame_t> pose_index_to_frame_;

  std::unordered_map<camera_t, size_t> camera_to_simple_radial_calib_index_;
  std::unordered_map<size_t, camera_t> simple_radial_calib_index_to_camera_;

  std::unordered_map<camera_t, size_t> camera_to_pinhole_calib_index_;
  std::unordered_map<size_t, camera_t> pinhole_calib_index_to_camera_;

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

  std::vector<unsigned int> pinhole_pose_indices_;
  std::vector<unsigned int> pinhole_calib_indices_;
  std::vector<unsigned int> pinhole_point_indices_;
  std::vector<float> pinhole_pixels_;

  std::vector<unsigned int> pinhole_fixed_pose_calib_indices_;
  std::vector<unsigned int> pinhole_fixed_pose_point_indices_;
  std::vector<float> pinhole_fixed_pose_poses_;
  std::vector<float> pinhole_fixed_pose_pixels_;

  std::vector<unsigned int> pinhole_fixed_point_pose_indices_;
  std::vector<unsigned int> pinhole_fixed_point_calib_indices_;
  std::vector<float> pinhole_fixed_point_points_;
  std::vector<float> pinhole_fixed_point_pixels_;
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