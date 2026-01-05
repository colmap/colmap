#include "colmap/estimators/caspar_bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <cstddef>
#include <iomanip>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "generated/solver.h"
#include "generated/solver_params.h"
#include <ceres/types.h>

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

    // Pass 1: Count observations
    for (const image_t image_id : config_.Images()) {
      CountImageObservations(image_id);
    }

    for (const auto point3D_id : config.VariablePoints()) {
      CountPointObservations(point3D_id);
    }

    for (const auto point3D_id : config.ConstantPoints()) {
      CountPointObservations(point3D_id);
    }

    // ApplyGaugeFixing();

    // Pass 2: Create factors and nodes
    for (const image_t image_id : config_.Images()) {
      AddImage(image_id);
    }

    for (const auto point3D_id : config.VariablePoints()) {
      AddPoint(point3D_id);
    }

    for (const auto point3D_id : config.ConstantPoints()) {
      AddPoint(point3D_id);
    }
  }

  void CountImageObservations(const image_t image_id) {
    const Image& image = reconstruction_.Image(image_id);
    if (!image.IsRefInFrame()) return;

    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }
      point3D_num_observations_[point2D.point3D_id] += 1;
    }
  }

  void CountPointObservations(const point3D_t point3D_id) {
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (!config_.HasImage(track_el.image_id)) {
        point3D_num_observations_[point3D_id] += 1;
      }
    }
  }

  void AddImage(const image_t image_id) {
    Image& image = reconstruction_.Image(image_id);

    if (image.CameraPtr()->model_id != CameraModelId::kSimpleRadial) {
      LOG(ERROR) << "ERROR! TRIED TO ADD NON SIMPLE RADIAL CAMERA!";
      return;
    }

    if (!image.IsRefInFrame()) {
      LOG(ERROR) << "ERROR! TRIED TO ADD NON TRIVIAL FRAME!!";
      return;
    }

    Camera& camera = *image.CameraPtr();
    Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();

    if (IsPoseVariable(image.FrameId())) {
      cam_from_world.rotation.normalize();
    }

    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }
      AddObservation(image, camera, cam_from_world, point2D);
    }
  }

  void AddPoint(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));

    if (gauge_fixed_points_.count(point3D_id)) {
      return;
    }

    Point3D& point3D = reconstruction_.Point3D(point3D_id);
    GetOrCreatePoint(point3D_id, point3D);

    size_t& num_observations = point3D_num_observations_[point3D_id];

    if (num_observations == point3D.track.Length()) {
      return;
    }

    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      num_observations += 1;

      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      if (!image.IsRefInFrame()) {
        continue;
      }

      AddSimpleRadialFixedCamFactor(image, camera, point2D, point3D);
      cameras_from_outside_config_.insert(camera.camera_id);
    }
  }

  void AddObservation(const Image& image,
                      const Camera& camera,
                      const Rigid3d& cam_from_world,
                      const Point2D& point2D) {
    const bool pose_var = IsPoseVariable(image.FrameId());
    const bool intrinsics_var = AreIntrinsicsVariable(camera.camera_id);
    const bool point_var = IsPointVariable(point2D.point3D_id);

    auto it = gauge_fixed_norm_frames_.find(image.FrameId());
    const bool is_norm_fixed = (it != gauge_fixed_norm_frames_.end());

    const Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);

    if (is_norm_fixed && pose_var) {
      const double norm_value = it->second;
      if (intrinsics_var && point_var) {
        AddSimpleRadialFixedTranslationNormFactor(
            image, camera, point2D, point3D, norm_value);
      } else {
        LOG(FATAL) << "Unsupported combination with fixed-norm frame";
      }
      return;
    }

    if (!pose_var && !intrinsics_var && !point_var) {
      return;
    }

    if (pose_var && intrinsics_var && point_var) {
      AddSimpleRadialFactor(image, camera, point2D, point3D);
    } else if (pose_var && intrinsics_var && !point_var) {
      AddSimpleRadialFixedPointFactor(image, camera, point2D, point3D);
    } else if (pose_var && !intrinsics_var && point_var) {
      AddSimpleRadialFixedIntrinsicsFactor(image, camera, point2D, point3D);
    } else if (pose_var && !intrinsics_var && !point_var) {
      AddSimpleRadialFixedIntrinsicsAndPointFactor(
          image, camera, point2D, point3D);
    } else if (!pose_var && intrinsics_var && point_var) {
      AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
    } else if (!pose_var && intrinsics_var && !point_var) {
      AddSimpleRadialFixedPoseAndPointFactor(image, camera, point2D, point3D);
    } else if (!pose_var && !intrinsics_var && point_var) {
      AddSimpleRadialFixedCamFactor(image, camera, point2D, point3D);
    }
  }

  void AddSimpleRadialFactor(const Image& image,
                             const Camera& camera,
                             const Point2D& point2D,
                             const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t camera_idx =
        GetOrCreateCamera(camera.camera_id, image.FrameId(), camera);
    simple_radial_camera_indices_.push_back(camera_idx);
    simple_radial_point_indices_.push_back(point_idx);
    simple_radial_pixels_.push_back(point2D.xy.x());
    simple_radial_pixels_.push_back(point2D.xy.y());
    num_simple_radial_++;
  }

  void AddSimpleRadialFixedIntrinsicsFactor(const Image& image,
                                            const Camera& camera,
                                            const Point2D& point2D,
                                            const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t pose_idx = GetOrCreatePose(
        camera.camera_id, image.FrameId(), image.FramePtr()->RigFromWorld());
    simple_radial_fixed_intrinsics_point_indices_.push_back(point_idx);
    simple_radial_fixed_intrinsics_pose_indices_.push_back(pose_idx);
    for (const auto& param : camera.params) {
      simple_radial_fixed_intrinsics_calibs_.push_back(param);
    }
    simple_radial_fixed_intrinsics_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_intrinsics_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_intrinsics_++;
  }

  void AddSimpleRadialFixedPoseFactor(const Image& image,
                                      const Camera& camera,
                                      const Point2D& point2D,
                                      const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t calib_idx = GetOrCreateCalib(camera.camera_id, camera);
    simple_radial_fixed_pose_point_indices_.push_back(point_idx);
    simple_radial_fixed_pose_calib_indices_.push_back(calib_idx);

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

  void AddSimpleRadialFixedCamFactor(const Image& image,
                                     const Camera& camera,
                                     const Point2D& point2D,
                                     const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    simple_radial_fixed_cam_point_indices_.push_back(point_idx);

    const Rigid3d& pose = image.FramePtr()->RigFromWorld();
    simple_radial_fixed_cam_cams_.push_back(pose.rotation.x());
    simple_radial_fixed_cam_cams_.push_back(pose.rotation.y());
    simple_radial_fixed_cam_cams_.push_back(pose.rotation.z());
    simple_radial_fixed_cam_cams_.push_back(pose.rotation.w());
    simple_radial_fixed_cam_cams_.push_back(pose.translation.x());
    simple_radial_fixed_cam_cams_.push_back(pose.translation.y());
    simple_radial_fixed_cam_cams_.push_back(pose.translation.z());

    for (const auto& param : camera.params) {
      simple_radial_fixed_cam_cams_.push_back(param);
    }

    simple_radial_fixed_cam_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_cam_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_cam_++;
  }

  void AddSimpleRadialFixedPointFactor(const Image& image,
                                       const Camera& camera,
                                       const Point2D& point2D,
                                       const Point3D& point3D) {
    const size_t camera_idx =
        GetOrCreateCamera(camera.camera_id, image.FrameId(), camera);
    simple_radial_fixed_point_cam_indices_.push_back(camera_idx);
    simple_radial_fixed_point_points_.push_back(point3D.xyz.x());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.y());
    simple_radial_fixed_point_points_.push_back(point3D.xyz.z());

    simple_radial_fixed_point_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_point_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_point_++;
  }

  void AddSimpleRadialFixedIntrinsicsAndPointFactor(const Image& image,
                                                    const Camera& camera,
                                                    const Point2D& point2D,
                                                    const Point3D& point3D) {
    const size_t pose_idx = GetOrCreatePose(
        camera.camera_id, image.FrameId(), image.FramePtr()->RigFromWorld());
    simple_radial_fixed_intrinsics_and_point_pose_indices_.push_back(pose_idx);

    for (const auto& param : camera.params) {
      simple_radial_fixed_intrinsics_and_point_calibs_.push_back(param);
    }

    simple_radial_fixed_intrinsics_and_point_points_.push_back(point3D.xyz.x());
    simple_radial_fixed_intrinsics_and_point_points_.push_back(point3D.xyz.y());
    simple_radial_fixed_intrinsics_and_point_points_.push_back(point3D.xyz.z());

    simple_radial_fixed_intrinsics_and_point_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_intrinsics_and_point_pixels_.push_back(point2D.xy.y());
    num_simple_radial_fixed_intrinsics_and_point_++;
  }

  void AddSimpleRadialFixedPoseAndPointFactor(const Image& image,
                                              const Camera& camera,
                                              const Point2D& point2D,
                                              const Point3D& point3D) {
    const size_t calib_idx = GetOrCreateCalib(camera.camera_id, camera);
    simple_radial_fixed_pose_and_point_calib_indices_.push_back(calib_idx);

    const Rigid3d& pose = image.FramePtr()->RigFromWorld();
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.rotation.x());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.rotation.y());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.rotation.z());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.rotation.w());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.translation.x());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.translation.y());
    simple_radial_fixed_pose_and_point_poses_.push_back(pose.translation.z());

    simple_radial_fixed_pose_and_point_points_.push_back(point3D.xyz.x());
    simple_radial_fixed_pose_and_point_points_.push_back(point3D.xyz.y());
    simple_radial_fixed_pose_and_point_points_.push_back(point3D.xyz.z());

    simple_radial_fixed_pose_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_pose_pixels_.push_back(point2D.xy.y());

    num_simple_radial_fixed_pose_and_point_++;
  };

  void AddSimpleRadialWithSeparateCalib(const Image& image,
                                        const Camera& camera,
                                        const Point2D& point2D,
                                        const Point3D& point3D) {
    const size_t pose_idx = GetOrCreatePose(
        camera.camera_id, image.FrameId(), image.FramePtr()->RigFromWorld());
    const size_t calib_idx = GetOrCreateCalib(camera.camera_id, camera);
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);

    simple_radial_with_separate_calib_pose_indices_.push_back(pose_idx);
    simple_radial_with_separate_calib_calib_indices_.push_back(calib_idx);
    simple_radial_with_separate_calib_point_indices_.push_back(point_idx);
    simple_radial_with_separate_calib_pixels_.push_back(point2D.xy.x());
    simple_radial_with_separate_calib_pixels_.push_back(point2D.xy.y());
    num_simple_radial_with_separate_calib_++;
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

  void AddSimpleRadialFixedTranslationNormFactor(const Image& image,
                                                 const Camera& camera,
                                                 const Point2D& point2D,
                                                 const Point3D& point3D,
                                                 const double norm_value) {
    const Rigid3d& pose = image.FramePtr()->RigFromWorld();

    const size_t rot3_idx = GetOrCreateRot3(image.FrameId(), pose);
    const size_t unit3_idx = GetOrCreateUnit3(image.FrameId(), pose);
    const size_t calib_idx = GetOrCreateCalib(camera.camera_id, camera);
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);

    simple_radial_fixed_translation_norm_rotation_indices_.push_back(rot3_idx);
    simple_radial_fixed_translation_norm_unit3_indices_.push_back(unit3_idx);
    simple_radial_fixed_translation_norm_calib_indices_.push_back(calib_idx);
    simple_radial_fixed_translation_norm_point_indices_.push_back(point_idx);
    simple_radial_fixed_translation_norm_values_.push_back(norm_value);
    simple_radial_fixed_translation_norm_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_translation_norm_pixels_.push_back(point2D.xy.y());

    num_simple_radial_fixed_translation_norm_++;
  }

  size_t GetOrCreatePose(const camera_t camera_id,
                         const frame_t frame_id,
                         const Rigid3d& pose) {
    auto [it, inserted] =
        frame_id_to_pose_index_.try_emplace(frame_id, num_poses_);
    if (inserted) {
      auto key = std::make_pair(frame_id, camera_id);
      pose_index_to_frame_id_[num_poses_] = key;
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

  size_t GetOrCreateCalib(const camera_t camera_id, const Camera& camera) {
    auto [it, inserted] =
        camera_id_to_calib_index_.try_emplace(camera_id, num_calibs_);
    if (inserted) {
      calib_index_to_camera_id_[num_calibs_] = camera_id;
      for (const auto& param : camera.params) {
        calib_data_.push_back(param);
      }
      num_calibs_++;
    }
    return it->second;
  }

  size_t GetOrCreateCamera(const camera_t camera_id,
                           const frame_t frame_id,
                           const Camera& camera) {
    auto key = std::make_pair(frame_id, camera_id);
    auto [it, inserted] = frame_camera_to_index_.try_emplace(key, num_cameras_);
    if (inserted) {
      index_to_frame_camera_[num_cameras_] = key;
      const Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      camera_data_.push_back(pose.rotation.x());
      camera_data_.push_back(pose.rotation.y());
      camera_data_.push_back(pose.rotation.z());
      camera_data_.push_back(pose.rotation.w());
      camera_data_.push_back(pose.translation.x());
      camera_data_.push_back(pose.translation.y());
      camera_data_.push_back(pose.translation.z());

      for (const auto& param : camera.params) {
        camera_data_.push_back(param);
      }
      num_cameras_++;
    }
    return it->second;
  }

  size_t GetOrCreateUnit3(const frame_t frame_id, const Rigid3d& pose) {
    auto [it, inserted] =
        frame_id_to_unit3_index_.try_emplace(frame_id, num_unit3_);
    if (inserted) {
      unit3_index_to_frame_id_[num_unit3_] = frame_id;
      Eigen::Vector3d direction = pose.translation.normalized();
      unit3_data_.push_back(direction.x());
      unit3_data_.push_back(direction.y());
      unit3_data_.push_back(direction.z());
      num_unit3_++;
    }
    return it->second;
  }

  size_t GetOrCreateRot3(const frame_t frame_id, const Rigid3d& pose) {
    auto [it, inserted] =
        frame_id_to_rot3_index_.try_emplace(frame_id, num_rot3_);
    if (inserted) {
      rot3_index_to_frame_id_[num_rot3_] = frame_id;
      rot3_data_.push_back(pose.rotation.x());
      rot3_data_.push_back(pose.rotation.y());
      rot3_data_.push_back(pose.rotation.z());
      rot3_data_.push_back(pose.rotation.w());
      num_rot3_++;
    }
    return it->second;
  }

  void ApplyGaugeFixing() {
    switch (config_.FixedGauge()) {
      case BundleAdjustmentGauge::UNSPECIFIED:
        break;
      case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
        FixGaugeWithTwoCamsFromWorld();
        break;
      case BundleAdjustmentGauge::THREE_POINTS:
        FixGaugeWithThreePoints();
        break;
      default:
        LOG(FATAL) << "Unknown BundleAdjustmentGauge";
    }
  }

  void FixGaugeWithTwoCamsFromWorld() {
    if (!options_.refine_rig_from_world) {
      return;
    }

    std::vector<std::pair<frame_t, camera_t>> variable_frames_in_problem;
    std::unordered_set<frame_t> frames_seen;

    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const frame_t frame_id = image.FrameId();

      if (frames_seen.count(frame_id)) continue;
      frames_seen.insert(frame_id);

      if (!image.FramePtr()->RigPtr()->IsRefSensor(
              image.CameraPtr()->SensorId())) {
        continue;
      }

      if (!config_.HasConstantRigFromWorldPose(frame_id)) {
        variable_frames_in_problem.push_back({frame_id, image.CameraId()});
      }
    }

    if (variable_frames_in_problem.size() < 2) {
      LOG(WARNING) << "Only " << variable_frames_in_problem.size()
                   << " variable frame(s) in problem. "
                   << "Falling back to THREE_POINTS gauge.";
      FixGaugeWithThreePoints();
      return;
    }

    auto [frame1_id, camera1_id] = variable_frames_in_problem[0];

    frame_t frame2_id = std::numeric_limits<frame_t>::max();
    camera_t camera2_id = std::numeric_limits<camera_t>::max();
    double best_baseline = 0.0;

    const Rigid3d& pose1 = reconstruction_.Frame(frame1_id).RigFromWorld();

    for (size_t i = 1; i < variable_frames_in_problem.size(); ++i) {
      auto [candidate_frame_id, candidate_camera_id] =
          variable_frames_in_problem[i];
      const Rigid3d& pose2 =
          reconstruction_.Frame(candidate_frame_id).RigFromWorld();
      const Eigen::Vector3d baseline = (pose1 * Inverse(pose2)).translation;
      const double baseline_magnitude = baseline.norm();

      // Check for Unit3 singularity (Skip if aligned with X)
      if (std::abs(baseline.normalized().x() - 1.0) < 1e-6) {
        continue;
      }

      if (baseline_magnitude > best_baseline) {
        best_baseline = baseline_magnitude;
        frame2_id = candidate_frame_id;
        camera2_id = candidate_camera_id;
      }
    }

    if (best_baseline <= 1e-9) {
      LOG(WARNING) << "Best baseline too small: " << best_baseline
                   << ". Falling back to THREE_POINTS gauge.";
      FixGaugeWithThreePoints();
      return;
    }

    config_.SetConstantRigFromWorldPose(frame1_id);

    const Rigid3d& pose2 = reconstruction_.Frame(frame2_id).RigFromWorld();
    const double translation_norm = pose2.translation.norm();

    if (translation_norm < 1e-6) {
      LOG(WARNING) << "Frame " << frame2_id
                   << " has near-zero translation norm: " << translation_norm
                   << ". Falling back to THREE_POINTS gauge fixing";
      FixGaugeWithThreePoints();
      return;
    }

    if (std::abs(pose2.translation.normalized().x() - 1.0) < 1e-6) {
      LOG(WARNING)
          << "Frame 2 translation near Unit3 singularity, consider "
             "different frame. Falling back to THREE_POINTS gauge fixing";
      FixGaugeWithThreePoints();
      return;
    }

    gauge_fixed_norm_frames_[frame2_id] = translation_norm;

    LOG(INFO) << "Fixed gauge with TWO_CAMS_FROM_WORLD:";
    LOG(INFO) << "  Frame " << frame1_id << " fully fixed (6 DOF)";
    LOG(INFO) << "  Frame " << frame2_id << " translation norm fixed (1 DOF)";
    LOG(INFO) << "  Fixed norm = " << translation_norm;
    LOG(INFO) << "  Baseline: " << best_baseline << " m";
  }

  void FixGaugeWithThreePoints() {
    FixedGaugeWithThreePoints fixed_gauge;

    std::vector<std::pair<size_t, point3D_t>> points_by_obs;
    for (const auto& [pid, num_obs] : point3D_num_observations_) {
      points_by_obs.push_back({num_obs, pid});
    }
    std::sort(points_by_obs.rbegin(), points_by_obs.rend());

    for (const auto& [num_obs, pid] : points_by_obs) {
      if (fixed_gauge.MaybeAddFixedPoint(reconstruction_.Point3D(pid).xyz)) {
        gauge_fixed_points_.insert(pid);
        if (fixed_gauge.num_fixed_points >= 3) break;
      }
    }

    for (const auto& [point_id, num_obs] : point3D_num_observations_) {
      if (!config_.HasConstantPoint(point_id)) {
        Point3D& point = reconstruction_.Point3D(point_id);
        if (fixed_gauge.MaybeAddFixedPoint(point.xyz)) {
          gauge_fixed_points_.insert(point_id);
          if (fixed_gauge.num_fixed_points >= 3) {
            return;
          }
        }
      }
    }

    LOG(WARNING) << "Failed to fix gauge with three points. "
                 << "Fixed " << fixed_gauge.num_fixed_points << " points.";
  }

  struct FixedGaugeWithThreePoints {
    Eigen::Index num_fixed_points = 0;
    Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();

    bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
      if (num_fixed_points >= 3) {
        return false;
      }
      fixed_points.col(num_fixed_points) = point;
      if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
        ++num_fixed_points;
        return true;
      } else {
        fixed_points.col(num_fixed_points).setZero();
        return false;
      }
    }
  };

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
    if (gauge_fixed_points_.count(point3D_id)) return false;
    if (config_.HasConstantPoint(point3D_id)) return false;
    const Point3D point3D = reconstruction_.Point3D(point3D_id);
    size_t num_obs_in_problem = point3D_num_observations_[point3D_id];
    if (point3D.track.Length() > num_obs_in_problem) return false;
    return true;
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    if (num_points_ > 0) {
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.set_Pose3_nodes_from_stacked_host(
          pose_data_.data(), 0, num_poses_);
    }
    if (num_rot3_ > 0) {
      solver.set_Rot3_nodes_from_stacked_host(rot3_data_.data(), 0, num_rot3_);
    }
    if (num_unit3_ > 0) {
      solver.set_Unit3_nodes_from_stacked_host(
          unit3_data_.data(), 0, num_unit3_);
    }
    if (num_calibs_ > 0) {
      solver.set_SimpleRadialCalib_nodes_from_stacked_host(
          calib_data_.data(), 0, num_calibs_);
    }
    if (num_cameras_ > 0) {
      solver.set_SimpleRadialCamera_nodes_from_stacked_host(
          camera_data_.data(), 0, num_cameras_);
    }

    if (num_simple_radial_ > 0) {
      solver.set_simple_radial_cam_indices_from_host(
          simple_radial_camera_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_point_indices_from_host(
          simple_radial_point_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_pixel_data_from_stacked_host(
          simple_radial_pixels_.data(), 0, num_simple_radial_);
    }

    if (num_simple_radial_fixed_intrinsics_ > 0) {
      solver.set_simple_radial_fixed_intrinsics_point_indices_from_host(
          simple_radial_fixed_intrinsics_point_indices_.data(),
          num_simple_radial_fixed_intrinsics_);
      solver.set_simple_radial_fixed_intrinsics_cam_T_world_indices_from_host(
          simple_radial_fixed_intrinsics_pose_indices_.data(),
          num_simple_radial_fixed_intrinsics_);
      solver
          .set_simple_radial_fixed_intrinsics_cam_calib_data_from_stacked_host(
              simple_radial_fixed_intrinsics_calibs_.data(),
              0,
              num_simple_radial_fixed_intrinsics_);
      solver.set_simple_radial_fixed_intrinsics_pixel_data_from_stacked_host(
          simple_radial_fixed_intrinsics_pixels_.data(),
          0,
          num_simple_radial_fixed_intrinsics_);
    }

    if (num_simple_radial_fixed_pose_ > 0) {
      solver.set_simple_radial_fixed_pose_point_indices_from_host(
          simple_radial_fixed_pose_point_indices_.data(),
          num_simple_radial_fixed_pose_);
      solver.set_simple_radial_fixed_pose_cam_calib_indices_from_host(
          simple_radial_fixed_pose_calib_indices_.data(),
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

    if (num_simple_radial_fixed_cam_ > 0) {
      solver.set_simple_radial_fixed_cam_point_indices_from_host(
          simple_radial_fixed_cam_point_indices_.data(),
          num_simple_radial_fixed_cam_);
      solver.set_simple_radial_fixed_cam_cam_data_from_stacked_host(
          simple_radial_fixed_cam_cams_.data(),
          0,
          num_simple_radial_fixed_cam_);
      solver.set_simple_radial_fixed_cam_pixel_data_from_stacked_host(
          simple_radial_fixed_cam_pixels_.data(),
          0,
          num_simple_radial_fixed_cam_);
    }

    if (num_simple_radial_fixed_point_ > 0) {
      solver.set_simple_radial_fixed_point_cam_indices_from_host(
          simple_radial_fixed_point_cam_indices_.data(),
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

    if (num_simple_radial_fixed_intrinsics_and_point_ > 0) {
      solver
          .set_simple_radial_fixed_intrinsics_and_point_cam_T_world_indices_from_host(
              simple_radial_fixed_intrinsics_and_point_pose_indices_.data(),
              num_simple_radial_fixed_intrinsics_and_point_);
      solver
          .set_simple_radial_fixed_intrinsics_and_point_cam_calib_data_from_stacked_host(
              simple_radial_fixed_intrinsics_and_point_calibs_.data(),
              0,
              num_simple_radial_fixed_intrinsics_and_point_);
      solver
          .set_simple_radial_fixed_intrinsics_and_point_point_data_from_stacked_host(
              simple_radial_fixed_intrinsics_and_point_points_.data(),
              0,
              num_simple_radial_fixed_intrinsics_and_point_);
      solver
          .set_simple_radial_fixed_intrinsics_and_point_pixel_data_from_stacked_host(
              simple_radial_fixed_intrinsics_and_point_pixels_.data(),
              0,
              num_simple_radial_fixed_intrinsics_and_point_);
    }

    if (num_simple_radial_fixed_pose_and_point_ > 0) {
      solver.set_simple_radial_fixed_pose_and_point_cam_calib_indices_from_host(
          simple_radial_fixed_pose_and_point_calib_indices_.data(),
          num_simple_radial_fixed_pose_and_point_);
      solver
          .set_simple_radial_fixed_pose_and_point_cam_T_world_data_from_stacked_host(
              simple_radial_fixed_pose_and_point_poses_.data(),
              0,
              num_simple_radial_fixed_pose_and_point_);
      solver
          .set_simple_radial_fixed_pose_and_point_point_data_from_stacked_host(
              simple_radial_fixed_pose_and_point_points_.data(),
              0,
              num_simple_radial_fixed_pose_and_point_);
      solver
          .set_simple_radial_fixed_pose_and_point_pixel_data_from_stacked_host(
              simple_radial_fixed_pose_and_point_pixels_.data(),
              0,
              num_simple_radial_fixed_pose_and_point_);
    }

    if (num_simple_radial_fixed_translation_norm_ > 0) {
      solver
          .set_simple_radial_fixed_translation_norm_rotation_indices_from_host(
              simple_radial_fixed_translation_norm_rotation_indices_.data(),
              num_simple_radial_fixed_translation_norm_);
      solver
          .set_simple_radial_fixed_translation_norm_translation_direction_indices_from_host(
              simple_radial_fixed_translation_norm_unit3_indices_.data(),
              num_simple_radial_fixed_translation_norm_);
      solver
          .set_simple_radial_fixed_translation_norm_cam_calib_indices_from_host(
              simple_radial_fixed_translation_norm_calib_indices_.data(),
              num_simple_radial_fixed_translation_norm_);
      solver.set_simple_radial_fixed_translation_norm_point_indices_from_host(
          simple_radial_fixed_translation_norm_point_indices_.data(),
          num_simple_radial_fixed_translation_norm_);
      solver
          .set_simple_radial_fixed_translation_norm_translation_norm_data_from_stacked_host(
              simple_radial_fixed_translation_norm_values_.data(),
              0,
              num_simple_radial_fixed_translation_norm_);
      solver
          .set_simple_radial_fixed_translation_norm_pixel_data_from_stacked_host(
              simple_radial_fixed_translation_norm_pixels_.data(),
              0,
              num_simple_radial_fixed_translation_norm_);
    }

    if (num_simple_radial_with_separate_calib_ > 0) {
      solver
          .set_simple_radial_with_separate_calib_cam_T_world_indices_from_host(
              simple_radial_with_separate_calib_pose_indices_.data(),
              num_simple_radial_with_separate_calib_);
      solver.set_simple_radial_with_separate_calib_cam_calib_indices_from_host(
          simple_radial_with_separate_calib_calib_indices_.data(),
          num_simple_radial_with_separate_calib_);
      solver.set_simple_radial_with_separate_calib_point_indices_from_host(
          simple_radial_with_separate_calib_point_indices_.data(),
          num_simple_radial_with_separate_calib_);
      solver.set_simple_radial_with_separate_calib_pixel_data_from_stacked_host(
          simple_radial_with_separate_calib_pixels_.data(),
          0,
          num_simple_radial_with_separate_calib_);
    }

    solver.finish_indices();
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0) {
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_poses_ > 0) {
      solver.get_Pose3_nodes_to_stacked_host(pose_data_.data(), 0, num_poses_);
    }
    if (num_cameras_ > 0) {
      solver.get_SimpleRadialCamera_nodes_to_stacked_host(
          camera_data_.data(), 0, num_cameras_);
    }
    if (num_calibs_ > 0) {
      solver.get_SimpleRadialCalib_nodes_to_stacked_host(
          calib_data_.data(), 0, num_calibs_);
    }
    if (num_rot3_ > 0) {
      solver.get_Rot3_nodes_to_stacked_host(rot3_data_.data(), 0, num_rot3_);
    }
    if (num_unit3_ > 0) {
      solver.get_Unit3_nodes_to_stacked_host(unit3_data_.data(), 0, num_unit3_);
    }
  }
  void WriteResultsToReconstruction() {
    // Write back points (gauge-fixed points are already filtered out)
    for (const auto& [idx, point_id] : index_to_point_id_) {
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

    // Write back Pose3 nodes (only variable frames)
    for (const auto& [idx, key] : pose_index_to_frame_id_) {
      const frame_t frame_id = key.first;
      if (!IsPoseVariable(frame_id)) continue;  // Skip constant frames

      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      const auto pose_stride = 7;
      pose.rotation.x() = pose_data_[idx * pose_stride + 0];
      pose.rotation.y() = pose_data_[idx * pose_stride + 1];
      pose.rotation.z() = pose_data_[idx * pose_stride + 2];
      pose.rotation.w() = pose_data_[idx * pose_stride + 3];
      pose.translation.x() = pose_data_[idx * pose_stride + 4];
      pose.translation.y() = pose_data_[idx * pose_stride + 5];
      pose.translation.z() = pose_data_[idx * pose_stride + 6];
    }

    // Write back intrinsics (only variable cameras)
    for (const auto& [idx, camera_id] : calib_index_to_camera_id_) {
      if (!AreIntrinsicsVariable(camera_id))
        continue;  // Skip constant intrinsics

      Camera& camera = reconstruction_.Camera(camera_id);
      for (size_t i = 0; i < camera.params.size(); i++) {
        camera.params[i] = calib_data_[idx * camera.params.size() + i];
      }
    }

    // Write back Camera nodes (pose + intrinsics bundled)
    for (const auto& [idx, key] : index_to_frame_camera_) {
      const frame_t frame_id = key.first;
      const camera_t camera_id = key.second;

      if (!IsPoseVariable(frame_id)) continue;  // Skip constant poses

      Camera& camera = reconstruction_.Camera(camera_id);
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();

      const auto camera_stride = 7 + camera.params.size();
      pose.rotation.x() = camera_data_[idx * camera_stride + 0];
      pose.rotation.y() = camera_data_[idx * camera_stride + 1];
      pose.rotation.z() = camera_data_[idx * camera_stride + 2];
      pose.rotation.w() = camera_data_[idx * camera_stride + 3];
      pose.translation.x() = camera_data_[idx * camera_stride + 4];
      pose.translation.y() = camera_data_[idx * camera_stride + 5];
      pose.translation.z() = camera_data_[idx * camera_stride + 6];

      if (AreIntrinsicsVariable(camera_id)) {
        for (size_t i = 0; i < camera.params.size(); i++) {
          camera.params[i] = camera_data_[idx * camera_stride + i + 7];
        }
      }
    }

    // Write back Rot3+Unit3 nodes (only norm-fixed frames, which are variable)
    for (const auto& [idx, frame_id] : rot3_index_to_frame_id_) {
      // Norm-fixed frames are by definition variable (just constrained)
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();

      pose.rotation.x() = rot3_data_[idx * 4 + 0];
      pose.rotation.y() = rot3_data_[idx * 4 + 1];
      pose.rotation.z() = rot3_data_[idx * 4 + 2];
      pose.rotation.w() = rot3_data_[idx * 4 + 3];
      pose.rotation.normalize();

      const size_t unit3_idx = frame_id_to_unit3_index_[frame_id];
      Eigen::Vector3d direction(unit3_data_[unit3_idx * 3 + 0],
                                unit3_data_[unit3_idx * 3 + 1],
                                unit3_data_[unit3_idx * 3 + 2]);
      direction.normalize();

      const double norm = gauge_fixed_norm_frames_[frame_id];
      pose.translation = direction * norm;
    }
  }

  size_t ComputeTotalResiduals() const {
    return 2 * (num_simple_radial_ + num_simple_radial_fixed_intrinsics_ +
                num_simple_radial_fixed_pose_ + num_simple_radial_fixed_cam_ +
                num_simple_radial_fixed_point_ +
                num_simple_radial_fixed_intrinsics_and_point_ +
                num_simple_radial_fixed_pose_and_point_ +
                num_simple_radial_with_separate_calib_);
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return dummy_problem_; }

  bool ValidateData() {
    if (num_points_ == 0 && num_calibs_ == 0 && num_cameras_ == 0 &&
        num_poses_ == 0) {
      LOG(WARNING) << "No data to optimize - likely pre-triangulation";
      return false;
    }

    size_t total_residuals = ComputeTotalResiduals();
    if (total_residuals == 0) {
      LOG(WARNING) << "No residuals to optimize";
      return false;
    }

    return true;
  }

  void ValidateConstantFramesHaveNoNodes() {
    // Collect all constant frame IDs
    std::unordered_set<frame_t> constant_frames;
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const frame_t frame_id = image.FrameId();
      if (!IsPoseVariable(frame_id)) {
        constant_frames.insert(frame_id);
      }
    }

    // Check that no constant frame appears in any variable node index
    for (const auto& [idx, key] : pose_index_to_frame_id_) {
      const frame_t frame_id = key.first;
      THROW_CHECK(!constant_frames.count(frame_id))
          << "Constant frame " << frame_id << " has Pose3 node!";
    }

    for (const auto& [idx, frame_id] : rot3_index_to_frame_id_) {
      THROW_CHECK(!constant_frames.count(frame_id))
          << "Constant frame " << frame_id << " has Rot3 node!";
    }

    for (const auto& [idx, key] : index_to_frame_camera_) {
      const frame_t frame_id = key.first;
      THROW_CHECK(!constant_frames.count(frame_id))
          << "Constant frame " << frame_id << " has Camera node!";
    }

    LOG(INFO) << "Validated: " << constant_frames.size()
              << " constant frames have no variable nodes";
  }

  ceres::Solver::Summary Solve() override {
    // FIRST THING: Save all constant state before ANY operations
    std::unordered_map<frame_t, Rigid3d> saved_constant_frame_poses;
    std::unordered_map<camera_t, std::vector<double>>
        saved_constant_camera_params;

    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const frame_t frame_id = image.FrameId();
      const camera_t camera_id = image.CameraId();

      if (!IsPoseVariable(frame_id)) {
        saved_constant_frame_poses[frame_id] = image.FramePtr()->RigFromWorld();
      }

      if (!AreIntrinsicsVariable(camera_id)) {
        saved_constant_camera_params[camera_id] = image.CameraPtr()->params;
      }
    }

    size_t total_residuals = ComputeTotalResiduals();

    if (!ValidateData()) {
      ceres::Solver::Summary summary;
      summary.termination_type = ceres::CONVERGENCE;
      summary.message = "Invalid data for optimization";
      return summary;
    }

    ValidateConstantFramesHaveNoNodes();

    caspar::GraphSolver solver =
        caspar::GraphSolver(params_,
                            num_points_,
                            num_poses_,
                            num_rot3_,
                            num_calibs_,
                            num_cameras_,
                            num_unit3_,
                            num_simple_radial_,
                            num_simple_radial_fixed_intrinsics_,
                            num_simple_radial_fixed_pose_,
                            num_simple_radial_fixed_cam_,
                            num_simple_radial_fixed_point_,
                            num_simple_radial_fixed_intrinsics_and_point_,
                            num_simple_radial_fixed_pose_and_point_,
                            num_simple_radial_fixed_translation_norm_,
                            num_simple_radial_with_separate_calib_);

    SetupSolverData(solver);
    const float result = solver.solve(false);
    ReadSolverResults(solver);
    WriteResultsToReconstruction();

    // Restore constant frames
    for (const auto& [frame_id, saved_pose] : saved_constant_frame_poses) {
      reconstruction_.Frame(frame_id).RigFromWorld() = saved_pose;

      // Verify restore actually happened
      const Rigid3d& restored_pose =
          reconstruction_.Frame(frame_id).RigFromWorld();
      THROW_CHECK(
          (restored_pose.rotation.coeffs() - saved_pose.rotation.coeffs())
              .norm() < 1e-15)
          << "Frame " << frame_id << " restore failed!";
      THROW_CHECK((restored_pose.translation - saved_pose.translation).norm() <
                  1e-15)
          << "Frame " << frame_id << " restore failed!";
    }

    ceres::Solver::Summary summary;
    summary.final_cost = result;
    summary.num_residuals_reduced = total_residuals;
    summary.termination_type = ceres::CONVERGENCE;
    return summary;
  }

 private:
  caspar::SolverParams params_;
  Reconstruction& reconstruction_;
  std::shared_ptr<ceres::Problem> dummy_problem_;

  std::unordered_map<frame_t, double> gauge_fixed_norm_frames_;
  std::unordered_set<point3D_t> gauge_fixed_points_;
  std::unordered_set<camera_t> cameras_from_outside_config_;

  size_t num_points_ = 0;
  size_t num_poses_ = 0;
  size_t num_calibs_ = 0;
  size_t num_cameras_ = 0;
  size_t num_unit3_ = 0;
  size_t num_rot3_ = 0;

  size_t num_simple_radial_ = 0;
  size_t num_simple_radial_fixed_intrinsics_ = 0;
  size_t num_simple_radial_fixed_pose_ = 0;
  size_t num_simple_radial_fixed_cam_ = 0;
  size_t num_simple_radial_fixed_point_ = 0;
  size_t num_simple_radial_fixed_intrinsics_and_point_ = 0;
  size_t num_simple_radial_fixed_pose_and_point_ = 0;
  size_t num_simple_radial_fixed_translation_norm_ = 0;
  size_t num_simple_radial_with_separate_calib_ = 0;

  std::vector<float> point_data_;
  std::vector<float> pose_data_;
  std::vector<float> calib_data_;
  std::vector<float> camera_data_;
  std::vector<float> unit3_data_;
  std::vector<float> rot3_data_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;
  std::unordered_map<frame_t, size_t> frame_id_to_pose_index_;
  std::unordered_map<size_t, std::pair<frame_t, camera_t>>
      pose_index_to_frame_id_;
  std::unordered_map<camera_t, size_t> camera_id_to_calib_index_;
  std::unordered_map<size_t, camera_t> calib_index_to_camera_id_;
  std::unordered_map<std::pair<frame_t, camera_t>, size_t>
      frame_camera_to_index_;
  std::unordered_map<size_t, std::pair<frame_t, camera_t>>
      index_to_frame_camera_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
  std::unordered_map<frame_t, size_t> frame_id_to_unit3_index_;
  std::unordered_map<size_t, frame_t> unit3_index_to_frame_id_;
  std::unordered_map<frame_t, size_t> frame_id_to_rot3_index_;
  std::unordered_map<size_t, frame_t> rot3_index_to_frame_id_;

  std::vector<unsigned int> simple_radial_camera_indices_;
  std::vector<unsigned int> simple_radial_point_indices_;
  std::vector<float> simple_radial_pixels_;

  std::vector<unsigned int> simple_radial_fixed_intrinsics_pose_indices_;
  std::vector<unsigned int> simple_radial_fixed_intrinsics_point_indices_;
  std::vector<float> simple_radial_fixed_intrinsics_calibs_;
  std::vector<float> simple_radial_fixed_intrinsics_pixels_;

  std::vector<unsigned int> simple_radial_fixed_pose_calib_indices_;
  std::vector<unsigned int> simple_radial_fixed_pose_point_indices_;
  std::vector<float> simple_radial_fixed_pose_poses_;
  std::vector<float> simple_radial_fixed_pose_pixels_;

  std::vector<unsigned int> simple_radial_fixed_cam_point_indices_;
  std::vector<float> simple_radial_fixed_cam_cams_;
  std::vector<float> simple_radial_fixed_cam_pixels_;

  std::vector<unsigned int> simple_radial_fixed_point_cam_indices_;
  std::vector<float> simple_radial_fixed_point_points_;
  std::vector<float> simple_radial_fixed_point_pixels_;

  std::vector<unsigned int>
      simple_radial_fixed_intrinsics_and_point_pose_indices_;
  std::vector<float> simple_radial_fixed_intrinsics_and_point_calibs_;
  std::vector<float> simple_radial_fixed_intrinsics_and_point_points_;
  std::vector<float> simple_radial_fixed_intrinsics_and_point_pixels_;

  std::vector<unsigned int> simple_radial_fixed_pose_and_point_calib_indices_;
  std::vector<float> simple_radial_fixed_pose_and_point_poses_;
  std::vector<float> simple_radial_fixed_pose_and_point_points_;
  std::vector<float> simple_radial_fixed_pose_and_point_pixels_;

  std::vector<unsigned int> simple_radial_scale_constraint_pose_indices_;
  std::vector<float> simple_radial_scale_constraint_points_;
  std::vector<float> simple_radial_scale_constraint_distance_constraints_;
  std::vector<float> simple_radial_scale_constraint_weights_;

  std::vector<unsigned int> simple_radial_with_separate_calib_pose_indices_;
  std::vector<unsigned int> simple_radial_with_separate_calib_calib_indices_;
  std::vector<unsigned int> simple_radial_with_separate_calib_point_indices_;
  std::vector<float> simple_radial_with_separate_calib_pixels_;

  std::vector<unsigned int>
      simple_radial_fixed_translation_norm_rotation_indices_;
  std::vector<unsigned int> simple_radial_fixed_translation_norm_unit3_indices_;
  std::vector<unsigned int> simple_radial_fixed_translation_norm_calib_indices_;
  std::vector<unsigned int> simple_radial_fixed_translation_norm_point_indices_;
  std::vector<float> simple_radial_fixed_translation_norm_values_;
  std::vector<float> simple_radial_fixed_translation_norm_pixels_;
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