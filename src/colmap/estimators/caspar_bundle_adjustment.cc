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

    ApplyGaugeFixing();
    DebugGaugeFixing("After gauge fixing");

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
    DebugGaugeFixing("After factor creation");
  }

  void CountImageObservations(const image_t image_id) {
    const Image& image = reconstruction_.Image(image_id);

    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
        continue;
      }
      point3D_num_observations_[point2D.point3D_id] += 1;
    }
  }

  void DebugGaugeFixing(const char* phase) {
    LOG(INFO) << "=== GAUGE DEBUG: " << phase << " ===";
    LOG(INFO) << "Norm-fixed frames: " << gauge_fixed_norm_frames_.size();
    for (const auto& [fid, norm] : gauge_fixed_norm_frames_) {
      LOG(INFO) << "  Frame " << fid << ": norm=" << norm;
    }
    LOG(INFO) << "CamFixedNorm nodes: " << num_cam_fixed_norm_;
    LOG(INFO) << "Fixed-norm factors: "
              << num_simple_radial_fixed_translation_norm_;
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

    if (config_.IsIgnoredPoint(point3D_id)) {
      return;  // Should never happen
    }
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

      AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
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

    if (!pose_var && !intrinsics_var && !point_var) {
      return;  // Nothing to optimise
    }

    // Handle norm-fixed frames with special factors
    if (is_norm_fixed && pose_var) {
      const double norm_value = it->second;

      if (intrinsics_var && point_var) {
        AddSimpleRadialFixedTranslationNormFactor(
            image, camera, point2D, point3D, norm_value);
      } else if (intrinsics_var && !point_var) {
        AddSimpleRadialFixedTranslationNormAndPointFactor(
            image, camera, point2D, point3D, norm_value);
      } else {
        // !intrinsics_var cases: not supported, fall back to fixed pose
        AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
      }
      return;
    }

    // Standard factors (non-norm-fixed)
    if (pose_var && intrinsics_var && point_var) {
      AddSimpleRadialFactor(image, camera, point2D, point3D);
    } else if (pose_var && intrinsics_var && !point_var) {
      AddSimpleRadialFixedPointFactor(image, camera, point2D, point3D);
    } else if (!pose_var && intrinsics_var && point_var) {
      AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
    } else {
      // Other combinations: use fixed pose factor as fallback
      AddSimpleRadialFixedPoseFactor(image, camera, point2D, point3D);
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

  void AddSimpleRadialFixedPoseFactor(const Image& image,
                                      const Camera& camera,
                                      const Point2D& point2D,
                                      const Point3D& point3D) {
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);
    const size_t cam_fixed_pose_idx =
        GetOrCreateCamFixedPose(camera.camera_id, camera);

    if (num_simple_radial_fixed_pose_ < 2) {  // Log first 2 factors
      LOG(INFO) << "=== AddSimpleRadialFixedPoseFactor #"
                << num_simple_radial_fixed_pose_ << " ===";
      LOG(INFO) << "  Frame: " << image.FrameId()
                << ", Camera: " << camera.camera_id;
      LOG(INFO) << "  cam_fixed_pose_idx: " << cam_fixed_pose_idx
                << ", point_idx: " << point_idx;
      LOG(INFO) << "  pixel: [" << point2D.xy.x() << ", " << point2D.xy.y()
                << "]";
      const Rigid3d& pose = image.FramePtr()->RigFromWorld();
      LOG(INFO) << "  pose quat: [" << pose.rotation.x() << ", "
                << pose.rotation.y() << ", " << pose.rotation.z() << ", "
                << pose.rotation.w() << "]";
      LOG(INFO) << "  pose trans: [" << pose.translation.x() << ", "
                << pose.translation.y() << ", " << pose.translation.z() << "]";
      LOG(INFO) << "  point3D: [" << point3D.xyz.x() << ", " << point3D.xyz.y()
                << ", " << point3D.xyz.z() << "]";
      LOG(INFO) << "  camera params: [" << camera.params[0] << ", "
                << camera.params[1] << ", " << camera.params[2] << ", "
                << camera.params[3] << "]";

      // Manually compute what the projection should be
      Eigen::Vector3d point_cam =
          pose.rotation * point3D.xyz + pose.translation;
      LOG(INFO) << "  point_cam: [" << point_cam.x() << ", " << point_cam.y()
                << ", " << point_cam.z() << "]";

      if (point_cam.z() > 0) {
        double x_norm = point_cam.x() / point_cam.z();
        double y_norm = point_cam.y() / point_cam.z();
        double r2 = x_norm * x_norm + y_norm * y_norm;
        double radial = 1.0 + camera.params[3] * r2;
        double u = camera.params[0] * x_norm * radial + camera.params[1];
        double v = camera.params[0] * y_norm * radial + camera.params[2];
        LOG(INFO) << "  projected: [" << u << ", " << v << "]";
        LOG(INFO) << "  residual: [" << (u - point2D.xy.x()) << ", "
                  << (v - point2D.xy.y()) << "]";
      } else {
        LOG(INFO) << "  point behind camera!";
      }
    }

    // CRITICAL: Push in the same order as we call the SET functions!
    // 1. cam_fixed_pose index first
    // 2. point index second
    simple_radial_fixed_pose_cam_fixed_pose_indices_.push_back(
        cam_fixed_pose_idx);
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

  void AddSimpleRadialFixedTranslationNormFactor(const Image& image,
                                                 const Camera& camera,
                                                 const Point2D& point2D,
                                                 const Point3D& point3D,
                                                 const double norm_value) {
    const Rigid3d& pose = image.FramePtr()->RigFromWorld();

    const size_t cam_fixed_norm_idx = GetOrCreateCamFixedNorm(
        camera.camera_id, image.FrameId(), pose, camera);
    const size_t point_idx = GetOrCreatePoint(point2D.point3D_id, point3D);

    simple_radial_fixed_translation_norm_cam_fixed_norm_indices_.push_back(
        cam_fixed_norm_idx);
    simple_radial_fixed_translation_norm_point_indices_.push_back(point_idx);
    simple_radial_fixed_translation_norm_values_.push_back(norm_value);
    simple_radial_fixed_translation_norm_pixels_.push_back(point2D.xy.x());
    simple_radial_fixed_translation_norm_pixels_.push_back(point2D.xy.y());

    num_simple_radial_fixed_translation_norm_++;
  }

  void AddSimpleRadialFixedTranslationNormAndPointFactor(
      const Image& image,
      const Camera& camera,
      const Point2D& point2D,
      const Point3D& point3D,
      const double norm_value) {
    const Rigid3d& pose = image.FramePtr()->RigFromWorld();

    const size_t cam_fixed_norm_idx = GetOrCreateCamFixedNorm(
        camera.camera_id, image.FrameId(), pose, camera);

    simple_radial_fixed_translation_norm_and_point_cam_fixed_norm_indices_
        .push_back(cam_fixed_norm_idx);
    simple_radial_fixed_translation_norm_and_point_norm_values_.push_back(
        norm_value);

    simple_radial_fixed_translation_norm_and_point_points_.push_back(
        point3D.xyz.x());
    simple_radial_fixed_translation_norm_and_point_points_.push_back(
        point3D.xyz.y());
    simple_radial_fixed_translation_norm_and_point_points_.push_back(
        point3D.xyz.z());

    simple_radial_fixed_translation_norm_and_point_pixels_.push_back(
        point2D.xy.x());
    simple_radial_fixed_translation_norm_and_point_pixels_.push_back(
        point2D.xy.y());

    num_simple_radial_fixed_translation_norm_and_point_++;
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

  size_t GetOrCreateCamFixedPose(const camera_t camera_id,
                                 const Camera& camera) {
    auto [it, inserted] = camera_id_to_cam_fixed_pose_index_.try_emplace(
        camera_id, num_cam_fixed_pose_);
    if (inserted) {
      LOG(INFO) << "Creating CamFixedPose[" << num_cam_fixed_pose_
                << "] for Camera " << camera_id
                << " with params: " << camera.params[0] << ", "
                << camera.params[1] << ", " << camera.params[2] << ", "
                << camera.params[3];
      cam_fixed_pose_index_to_camera_id_[num_cam_fixed_pose_] = camera_id;
      for (const auto& param : camera.params) {
        cam_fixed_pose_data_.push_back(param);
      }
      num_cam_fixed_pose_++;
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

  size_t GetOrCreateCamFixedNorm(const camera_t camera_id,
                                 const frame_t frame_id,
                                 const Rigid3d& pose,
                                 const Camera& camera) {
    auto key = std::make_pair(frame_id, camera_id);
    auto [it, inserted] = frame_camera_to_cam_fixed_norm_index_.try_emplace(
        key, num_cam_fixed_norm_);
    if (inserted) {
      cam_fixed_norm_index_to_frame_camera_[num_cam_fixed_norm_] = key;

      // Rotation (4 floats: x, y, z, w)
      cam_fixed_norm_data_.push_back(pose.rotation.x());
      cam_fixed_norm_data_.push_back(pose.rotation.y());
      cam_fixed_norm_data_.push_back(pose.rotation.z());
      cam_fixed_norm_data_.push_back(pose.rotation.w());

      // Translation direction (3 floats: normalized)
      Eigen::Vector3d direction = pose.translation.normalized();
      cam_fixed_norm_data_.push_back(direction.x());
      cam_fixed_norm_data_.push_back(direction.y());
      cam_fixed_norm_data_.push_back(direction.z());

      // Calibration (4 floats: f, cx, cy, k)
      for (const auto& param : camera.params) {
        cam_fixed_norm_data_.push_back(param);
      }

      num_cam_fixed_norm_++;
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

    // Collect variable frames
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

    // Select two frames with good baseline
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

    gauge_fixed_fully_constant_frame_ = frame1_id;

    const Rigid3d& pose2 = reconstruction_.Frame(frame2_id).RigFromWorld();
    const double translation_norm = pose2.translation.norm();

    if (translation_norm < 1e-6) {
      LOG(WARNING) << "Frame " << frame2_id
                   << " has near-zero translation norm: " << translation_norm
                   << ". Falling back to THREE_POINTS gauge fixing";
      gauge_fixed_fully_constant_frame_ = kInvalidFrameId;
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

    auto IsEffectivelyConstant = [&](point3D_t point3D_id,
                                     size_t num_observations) {
      if (config_.HasConstantPoint(point3D_id)) return true;
      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      return point3D.track.Length() > num_observations;
    };

    // Convert to vector and sort by point ID for deterministic iteration
    std::vector<std::pair<point3D_t, size_t>> sorted_points(
        point3D_num_observations_.begin(), point3D_num_observations_.end());
    std::sort(sorted_points.begin(), sorted_points.end());

    // First loop
    for (const auto& [point3D_id, num_observations] : sorted_points) {
      if (!IsEffectivelyConstant(point3D_id, num_observations)) continue;
      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      if (fixed_gauge.MaybeAddFixedPoint(point3D.xyz)) {
        gauge_fixed_points_.insert(point3D_id);
        if (fixed_gauge.num_fixed_points >= 3) return;
      }
    }

    // Second loop
    for (const auto& [point3D_id, num_observations] : sorted_points) {
      if (IsEffectivelyConstant(point3D_id, num_observations)) continue;
      Point3D& point3D = reconstruction_.Point3D(point3D_id);
      if (fixed_gauge.MaybeAddFixedPoint(point3D.xyz)) {
        gauge_fixed_points_.insert(point3D_id);
        if (fixed_gauge.num_fixed_points >= 3) return;
      }
    }

    LOG(WARNING) << "Failed to fix Gauge...";
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
    if (frame_id == gauge_fixed_fully_constant_frame_) return false;
    return true;
  }

  bool AreIntrinsicsVariable(const camera_t camera_id) {
    // We only support variable intrinsics
    return true;

    // bool any_refinement = options_.refine_focal_length ||
    //                       options_.refine_principal_point ||
    //                       options_.refine_extra_params;
    // if (!any_refinement) return false;
    // if (config_.HasConstantCamIntrinsics(camera_id)) return false;
    // if (cameras_from_outside_config_.count(camera_id)) return false;
    // return true;
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
    LOG(INFO) << "=== CASPAR SOLVER SETUP ===";
    LOG(INFO) << "Node counts (in definition order):";
    LOG(INFO) << "  1. Point: " << num_points_;
    LOG(INFO) << "  2. SimpleRadialCamera (Pose3+Calib): " << num_cameras_;
    LOG(INFO) << "  3. SimpleRadialCameraFixedPose (Calib only): "
              << num_cam_fixed_pose_;
    LOG(INFO)
        << "  4. SimpleRadialCameraFixedTranslationNorm (Rot3+Unit3+Calib): "
        << num_cam_fixed_norm_;

    LOG(INFO) << "Gauge-fixed elements:";
    LOG(INFO) << "  Fixed points: " << gauge_fixed_points_.size();
    if (!gauge_fixed_points_.empty()) {
      for (auto pid : gauge_fixed_points_) {
        const Point3D& p = reconstruction_.Point3D(pid);
        LOG(INFO) << "    Point " << pid << ": [" << p.xyz.transpose() << "]";
      }
    }

    LOG(INFO) << "  Fixed-norm frames: " << gauge_fixed_norm_frames_.size();
    if (!gauge_fixed_norm_frames_.empty()) {
      for (const auto& [fid, norm] : gauge_fixed_norm_frames_) {
        const Rigid3d& pose = reconstruction_.Frame(fid).RigFromWorld();
        LOG(INFO) << "    Frame " << fid << " norm=" << norm << " pos=["
                  << pose.translation.transpose() << "]";
      }
    }

    LOG(INFO) << "Factor counts (observations):";
    LOG(INFO) << "  simple_radial: " << num_simple_radial_;
    LOG(INFO) << "  fixed_pose: " << num_simple_radial_fixed_pose_;
    LOG(INFO) << "  fixed_point: " << num_simple_radial_fixed_point_;
    LOG(INFO) << "  fixed_translation_norm: "
              << num_simple_radial_fixed_translation_norm_;
    LOG(INFO) << "  fixed_translation_norm_and_point: "
              << num_simple_radial_fixed_translation_norm_and_point_;

    size_t total_factors = num_simple_radial_ + num_simple_radial_fixed_pose_ +
                           num_simple_radial_fixed_point_ +
                           num_simple_radial_fixed_translation_norm_ +
                           num_simple_radial_fixed_translation_norm_and_point_;

    size_t total_residuals = ComputeTotalResiduals();

    LOG(INFO) << "Total factors: " << total_factors;
    LOG(INFO) << "Total residuals: " << total_residuals << " (2 per factor)";
    LOG(INFO) << "===========================";

    // Set node data (must match order in GraphSolver constructor!)
    // Order: Point, SimpleRadialCamera, SimpleRadialCameraFixedPose,
    // SimpleRadialCameraFixedTranslationNorm
    if (num_points_ > 0) {
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    }
    if (num_cameras_ > 0) {
      solver.set_SimpleRadialCamera_nodes_from_stacked_host(
          camera_data_.data(), 0, num_cameras_);
    }
    if (num_cam_fixed_pose_ > 0) {
      LOG(INFO) << "=== CAMFIXEDPOSE NODE SETUP ===";
      LOG(INFO) << "num_cam_fixed_pose_: " << num_cam_fixed_pose_;
      LOG(INFO) << "cam_fixed_pose_data_ size: " << cam_fixed_pose_data_.size();
      LOG(INFO) << "First node data: [" << cam_fixed_pose_data_[0] << ", "
                << cam_fixed_pose_data_[1] << ", " << cam_fixed_pose_data_[2]
                << ", " << cam_fixed_pose_data_[3] << "]";

      // Check factor connectivity
      LOG(INFO) << "num_simple_radial_fixed_pose_: "
                << num_simple_radial_fixed_pose_;
      if (num_simple_radial_fixed_pose_ > 0) {
        LOG(INFO) << "First factor references:";
        LOG(INFO) << "  cam_fixed_pose_idx: "
                  << simple_radial_fixed_pose_cam_fixed_pose_indices_[0];
        LOG(INFO) << "  point_idx: "
                  << simple_radial_fixed_pose_point_indices_[0];
      }

      solver.set_SimpleRadialCameraFixedPose_nodes_from_stacked_host(
          cam_fixed_pose_data_.data(), 0, num_cam_fixed_pose_);
    }
    if (num_cam_fixed_norm_ > 0) {
      solver.set_SimpleRadialCameraFixedTranslationNorm_nodes_from_stacked_host(
          cam_fixed_norm_data_.data(), 0, num_cam_fixed_norm_);
    }

    if (num_simple_radial_ > 0) {
      solver.set_simple_radial_cam_indices_from_host(
          simple_radial_camera_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_point_indices_from_host(
          simple_radial_point_indices_.data(), num_simple_radial_);
      solver.set_simple_radial_pixel_data_from_stacked_host(
          simple_radial_pixels_.data(), 0, num_simple_radial_);
    }

    if (num_simple_radial_fixed_pose_ > 0) {
      solver.set_simple_radial_fixed_pose_cam_fixed_pose_indices_from_host(
          simple_radial_fixed_pose_cam_fixed_pose_indices_.data(),
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

    if (num_simple_radial_fixed_translation_norm_ > 0) {
      solver
          .set_simple_radial_fixed_translation_norm_cam_fixed_norm_indices_from_host(
              simple_radial_fixed_translation_norm_cam_fixed_norm_indices_
                  .data(),
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

    if (num_simple_radial_fixed_translation_norm_and_point_ > 0) {
      solver.set_simple_radial_fixed_translation_norm_and_point_cam_fixed_norm_indices_from_host(
          simple_radial_fixed_translation_norm_and_point_cam_fixed_norm_indices_
              .data(),
          num_simple_radial_fixed_translation_norm_and_point_);
      solver
          .set_simple_radial_fixed_translation_norm_and_point_translation_norm_data_from_stacked_host(
              simple_radial_fixed_translation_norm_and_point_norm_values_
                  .data(),
              0,
              num_simple_radial_fixed_translation_norm_and_point_);
      solver
          .set_simple_radial_fixed_translation_norm_and_point_point_data_from_stacked_host(
              simple_radial_fixed_translation_norm_and_point_points_.data(),
              0,
              num_simple_radial_fixed_translation_norm_and_point_);
      solver
          .set_simple_radial_fixed_translation_norm_and_point_pixel_data_from_stacked_host(
              simple_radial_fixed_translation_norm_and_point_pixels_.data(),
              0,
              num_simple_radial_fixed_translation_norm_and_point_);
    }

    LOG(INFO) << "=== SETTING FACTOR COUNTS ===";
    LOG(INFO) << "  simple_radial: " << num_simple_radial_;
    LOG(INFO) << "  simple_radial_fixed_pose: "
              << num_simple_radial_fixed_pose_;
    LOG(INFO) << "  simple_radial_fixed_point: "
              << num_simple_radial_fixed_point_;
    LOG(INFO) << "  simple_radial_fixed_translation_norm: "
              << num_simple_radial_fixed_translation_norm_;
    LOG(INFO) << "  simple_radial_fixed_translation_norm_and_point: "
              << num_simple_radial_fixed_translation_norm_and_point_;

    // CRITICAL: Tell solver how many factors of each type we have!
    solver.set_simple_radial_num(num_simple_radial_);
    solver.set_simple_radial_fixed_pose_num(num_simple_radial_fixed_pose_);
    solver.set_simple_radial_fixed_point_num(num_simple_radial_fixed_point_);
    solver.set_simple_radial_fixed_translation_norm_num(
        num_simple_radial_fixed_translation_norm_);
    solver.set_simple_radial_fixed_translation_norm_and_point_num(
        num_simple_radial_fixed_translation_norm_and_point_);

    LOG(INFO) << "=== CALLING finish_indices() ===";
    try {
      solver.finish_indices();
      LOG(INFO) << "finish_indices() completed successfully";
    } catch (const std::exception& e) {
      LOG(ERROR) << "finish_indices() threw exception: " << e.what();
      throw;
    }

    LOG(INFO) << "=== SOLVER PARAMETERS ===";
    LOG(INFO) << "max_solver_iters: " << params_.solver_iter_max;
    LOG(INFO) << "max_pcg_iters: " << params_.pcg_iter_max;
    LOG(INFO) << "Finished setting indices, ready to solve";
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    LOG(INFO) << "=== READING SOLVER RESULTS ===";

    // Read nodes in same order as they were set
    if (num_points_ > 0) {
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);
      LOG(INFO) << "Read " << num_points_ << " Point nodes";
    }
    if (num_cameras_ > 0) {
      solver.get_SimpleRadialCamera_nodes_to_stacked_host(
          camera_data_.data(), 0, num_cameras_);
      LOG(INFO) << "Read " << num_cameras_ << " SimpleRadialCamera nodes";
    }
    if (num_cam_fixed_pose_ > 0) {
      LOG(INFO) << "Reading " << num_cam_fixed_pose_
                << " CamFixedPose nodes...";
      LOG(INFO) << "  Before read: " << cam_fixed_pose_data_[0] << ", "
                << cam_fixed_pose_data_[1] << ", " << cam_fixed_pose_data_[2]
                << ", " << cam_fixed_pose_data_[3];

      solver.get_SimpleRadialCameraFixedPose_nodes_to_stacked_host(
          cam_fixed_pose_data_.data(), 0, num_cam_fixed_pose_);

      LOG(INFO) << "  After read: " << cam_fixed_pose_data_[0] << ", "
                << cam_fixed_pose_data_[1] << ", " << cam_fixed_pose_data_[2]
                << ", " << cam_fixed_pose_data_[3];
    }
    if (num_cam_fixed_norm_ > 0) {
      solver.get_SimpleRadialCameraFixedTranslationNorm_nodes_to_stacked_host(
          cam_fixed_norm_data_.data(), 0, num_cam_fixed_norm_);
      LOG(INFO) << "Read " << num_cam_fixed_norm_ << " CamFixedNorm nodes";
    }

    LOG(INFO) << "=== FINISHED READING SOLVER RESULTS ===";
  }

  void WriteResultsToReconstruction() {
    LOG(INFO) << "=== WRITE RESULTS TO RECONSTRUCTION ===";
    LOG(INFO) << "Writing " << index_to_point_id_.size() << " points";
    LOG(INFO) << "Writing " << cam_fixed_pose_index_to_camera_id_.size()
              << " CamFixedPose nodes";
    LOG(INFO) << "Writing " << index_to_frame_camera_.size() << " Camera nodes";
    LOG(INFO) << "Writing " << cam_fixed_norm_index_to_frame_camera_.size()
              << " CamFixedNorm nodes";

    // Write back points (gauge-fixed points are already filtered out)
    for (const auto& [idx, point_id] : index_to_point_id_) {
      if (!IsPointVariable(point_id)) {
        LOG(WARNING) << "Tried to write to constant point";
        continue;
      }
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

    // Write back CamFixedPose nodes (just calibration)
    for (const auto& [idx, camera_id] : cam_fixed_pose_index_to_camera_id_) {
      LOG(INFO) << "Writing CamFixedPose[" << idx << "] to Camera " << camera_id
                << ", intrinsics_var=" << AreIntrinsicsVariable(camera_id);

      if (!AreIntrinsicsVariable(camera_id))
        continue;  // Skip constant intrinsics

      Camera& camera = reconstruction_.Camera(camera_id);
      LOG(INFO) << "  Before: " << camera.params[0] << ", " << camera.params[1]
                << ", " << camera.params[2] << ", " << camera.params[3];

      for (size_t i = 0; i < camera.params.size(); i++) {
        camera.params[i] = cam_fixed_pose_data_[idx * camera.params.size() + i];
      }

      LOG(INFO) << "  After: " << camera.params[0] << ", " << camera.params[1]
                << ", " << camera.params[2] << ", " << camera.params[3];
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

    // Write back CamFixedNorm nodes (rotation + direction + calibration)
    for (const auto& [idx, key] : cam_fixed_norm_index_to_frame_camera_) {
      const frame_t frame_id = key.first;
      const camera_t camera_id = key.second;

      // Safety check: only write back if pose is actually variable
      // (norm-fixed frames should be variable, just constrained)
      if (!IsPoseVariable(frame_id)) {
        LOG(WARNING) << "Skipping CamFixedNorm write-back for constant frame "
                     << frame_id << " (this shouldn't happen!)";

        // Still write back calibration if variable
        if (AreIntrinsicsVariable(camera_id)) {
          Camera& camera = reconstruction_.Camera(camera_id);
          const size_t stride = 11;
          for (size_t i = 0; i < camera.params.size(); i++) {
            camera.params[i] = cam_fixed_norm_data_[idx * stride + 7 + i];
          }
        }
        continue;
      }

      Camera& camera = reconstruction_.Camera(camera_id);
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();

      const size_t stride = 11;  // 4 (quat) + 3 (unit3) + 4 (calib)

      // Read rotation (quaternion)
      pose.rotation.x() = cam_fixed_norm_data_[idx * stride + 0];
      pose.rotation.y() = cam_fixed_norm_data_[idx * stride + 1];
      pose.rotation.z() = cam_fixed_norm_data_[idx * stride + 2];
      pose.rotation.w() = cam_fixed_norm_data_[idx * stride + 3];

      // Read translation direction (Unit3)
      Eigen::Vector3d direction(cam_fixed_norm_data_[idx * stride + 4],
                                cam_fixed_norm_data_[idx * stride + 5],
                                cam_fixed_norm_data_[idx * stride + 6]);

      // Reconstruct translation with fixed norm
      const double norm = gauge_fixed_norm_frames_[frame_id];
      pose.translation = direction * norm;

      // Read calibration
      if (AreIntrinsicsVariable(camera_id)) {
        for (size_t i = 0; i < camera.params.size(); i++) {
          camera.params[i] = cam_fixed_norm_data_[idx * stride + 7 + i];
        }
      }
    }
  }

  size_t ComputeTotalResiduals() const {
    return 2 * (num_simple_radial_ + num_simple_radial_fixed_pose_ +
                num_simple_radial_fixed_point_ +
                num_simple_radial_fixed_translation_norm_ +
                num_simple_radial_fixed_translation_norm_and_point_);
  }

  std::shared_ptr<ceres::Problem>& Problem() override { return dummy_problem_; }

  bool ValidateData() {
    if (num_points_ == 0 && num_cam_fixed_pose_ == 0 && num_cameras_ == 0 &&
        num_cam_fixed_norm_ == 0) {
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
    for (const auto& [idx, key] : cam_fixed_norm_index_to_frame_camera_) {
      const frame_t frame_id = key.first;
      THROW_CHECK(!constant_frames.count(frame_id))
          << "Constant frame " << frame_id << " has CamFixedNorm node!";
    }

    for (const auto& [idx, key] : index_to_frame_camera_) {
      const frame_t frame_id = key.first;
      THROW_CHECK(!constant_frames.count(frame_id))
          << "Constant frame " << frame_id << " has Camera node!";
    }
  }

  ceres::Solver::Summary Solve() override {
    size_t total_residuals = ComputeTotalResiduals();

    if (!ValidateData()) {
      ceres::Solver::Summary summary;
      summary.termination_type = ceres::CONVERGENCE;
      summary.message = "Invalid data for optimization";
      return summary;
    }

    ValidateConstantFramesHaveNoNodes();

    // Compute scene scale before optimization
    auto ComputeSceneScale = [&]() {
      double total_dist = 0;
      int count = 0;
      for (const auto& [idx, point_id] : index_to_point_id_) {
        if (count > 0) {
          const Point3D& p = reconstruction_.Point3D(point_id);
          total_dist += p.xyz.norm();
        }
        if (++count >= 100) break;  // Sample first 100 points
      }
      return total_dist / count + 1e-5;
    };

    double scale_before = ComputeSceneScale();
    LOG(INFO) << "Scene scale BEFORE: " << scale_before;

    caspar::GraphSolver solver = caspar::GraphSolver(
        params_,
        // Node counts (must match Python definition order!)
        num_points_,          // Point
        num_cameras_,         // SimpleRadialCamera
        num_cam_fixed_pose_,  // SimpleRadialCameraFixedPose
        num_cam_fixed_norm_,  // SimpleRadialCameraFixedTranslationNorm
        // Factor counts
        num_simple_radial_,
        num_simple_radial_fixed_pose_,
        num_simple_radial_fixed_point_,
        num_simple_radial_fixed_translation_norm_,
        num_simple_radial_fixed_translation_norm_and_point_);

    SetupSolverData(solver);

    LOG(INFO) << "=== CALLING CASPAR SOLVER ===";
    LOG(INFO) << "Starting solve...";
    const float result = solver.solve(false);  // Enable progress printing
    LOG(INFO) << "Solve completed with cost: " << result;

    double scale_after = ComputeSceneScale();
    LOG(INFO) << "Scene scale AFTER: " << scale_after;
    LOG(INFO) << "Scale ratio: " << scale_after / scale_before;
    LOG(INFO) << "Cost AFTER: " << result;

    if (std::isnan(result) || std::isinf(result) || result > 1e10) {
      LOG(ERROR) << "Solver returned abnormal cost: " << result;
      LOG(ERROR) << "This usually indicates a problem with the setup";
    }

    ReadSolverResults(solver);
    WriteResultsToReconstruction();

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

  BundleAdjustmentGauge gauge_;
  std::unordered_map<frame_t, double> gauge_fixed_norm_frames_;
  std::unordered_set<point3D_t> gauge_fixed_points_;
  frame_t gauge_fixed_fully_constant_frame_ = kInvalidFrameId;
  std::unordered_set<camera_t> cameras_from_outside_config_;

  size_t num_points_ = 0;
  size_t num_cam_fixed_pose_ = 0;  // SimpleRadialCameraFixedPose nodes
  size_t num_cameras_ = 0;
  size_t num_cam_fixed_norm_ =
      0;  // SimpleRadialCameraFixedTranslationNorm nodes

  size_t num_simple_radial_ = 0;
  size_t num_simple_radial_fixed_pose_ = 0;
  size_t num_simple_radial_fixed_point_ = 0;
  size_t num_simple_radial_fixed_translation_norm_ = 0;
  size_t num_simple_radial_fixed_translation_norm_and_point_ = 0;

  std::vector<float> point_data_;
  std::vector<float> cam_fixed_pose_data_;  // Just calibration (4 params)
  std::vector<float> camera_data_;  // Pose3 + calibration (7 + 4 params)
  std::vector<float>
      cam_fixed_norm_data_;  // Rot3 + Unit3 + calibration (4 + 3 + 4 params)

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;
  std::unordered_map<camera_t, size_t> camera_id_to_cam_fixed_pose_index_;
  std::unordered_map<size_t, camera_t> cam_fixed_pose_index_to_camera_id_;
  std::unordered_map<std::pair<frame_t, camera_t>, size_t>
      frame_camera_to_index_;
  std::unordered_map<size_t, std::pair<frame_t, camera_t>>
      index_to_frame_camera_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
  std::unordered_map<std::pair<frame_t, camera_t>, size_t>
      frame_camera_to_cam_fixed_norm_index_;
  std::unordered_map<size_t, std::pair<frame_t, camera_t>>
      cam_fixed_norm_index_to_frame_camera_;

  std::vector<unsigned int> simple_radial_camera_indices_;
  std::vector<unsigned int> simple_radial_point_indices_;
  std::vector<float> simple_radial_pixels_;

  std::vector<unsigned int> simple_radial_fixed_pose_cam_fixed_pose_indices_;
  std::vector<unsigned int> simple_radial_fixed_pose_point_indices_;
  std::vector<float> simple_radial_fixed_pose_poses_;  // Constant pose data
  std::vector<float> simple_radial_fixed_pose_pixels_;

  std::vector<unsigned int> simple_radial_fixed_point_cam_indices_;
  std::vector<float> simple_radial_fixed_point_points_;
  std::vector<float> simple_radial_fixed_point_pixels_;

  std::vector<unsigned int>
      simple_radial_fixed_translation_norm_cam_fixed_norm_indices_;
  std::vector<unsigned int> simple_radial_fixed_translation_norm_point_indices_;
  std::vector<float> simple_radial_fixed_translation_norm_values_;
  std::vector<float> simple_radial_fixed_translation_norm_pixels_;

  std::vector<unsigned int>
      simple_radial_fixed_translation_norm_and_point_cam_fixed_norm_indices_;
  std::vector<float>
      simple_radial_fixed_translation_norm_and_point_norm_values_;
  std::vector<float> simple_radial_fixed_translation_norm_and_point_points_;
  std::vector<float> simple_radial_fixed_translation_norm_and_point_pixels_;
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