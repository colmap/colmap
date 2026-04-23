#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/sensor/models.h"
#ifdef CASPAR_ENABLED
#include "caspar/caspar_model_adapter.h"
#include <solver.h>
#endif

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
                       Reconstruction& reconstruction)
      : BundleAdjuster(options, config), reconstruction_(reconstruction) {
    VLOG(1) << "Using Caspar bundle adjuster";

    BuildObservationCounts();
    BuildCameraFrameIndex();
    BuildFactors();
  }

 private:
  // Returns the adapter for a given model ID, creating it if needed.
  // Returns nullptr for unsupported models.
  ICasparModelAdapter* GetAdapter(const CameraModelId model_id) {
    auto it = adapters_.find(model_id);
    if (it != adapters_.end()) return it->second.get();

    auto adapter = CreateCasparAdapter(model_id);
    if (!adapter) return nullptr;

    auto* ptr = adapter.get();
    adapters_[model_id] = std::move(adapter);
    model_data_per_model_.emplace(model_id, ModelData{});
    calib_num_per_model_[model_id] = 0;
    return ptr;
  }

  void BuildCameraFrameIndex() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();
      if (!GetAdapter(camera.model_id)) continue;
      auto key = std::make_pair(image.CameraId(), image.FrameId());
      camera_frame_to_images_[key].push_back(image_id);
    }
  }

  void BuildObservationCounts() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();
      if (!GetAdapter(camera.model_id)) {
        LOG(WARNING) << "Skipping image " << image_id
                     << " with unsupported camera model: "
                     << camera.ModelName();
        continue;
      }
      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id))
          continue;
        point3D_num_observations_[point2D.point3D_id]++;
      }
    }
    for (const auto point3D_id : config_.VariablePoints())
      CountExternalObservations(point3D_id);
    for (const auto point3D_id : config_.ConstantPoints())
      CountExternalObservations(point3D_id);
  }

  void CountExternalObservations(const point3D_t point3D_id) {
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (!config_.HasImage(track_el.image_id)) {
        Image& image = reconstruction_.Image(track_el.image_id);
        Camera& camera = *image.CameraPtr();
        if (GetAdapter(camera.model_id))
          point3D_num_observations_[point3D_id]++;
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
    for (const image_t image_id : config_.Images())
      sorted_camera_ids.push_back(reconstruction_.Image(image_id).CameraId());
    std::sort(sorted_camera_ids.begin(), sorted_camera_ids.end());
    sorted_camera_ids.erase(
        std::unique(sorted_camera_ids.begin(), sorted_camera_ids.end()),
        sorted_camera_ids.end());

    for (const camera_t camera_id : sorted_camera_ids) {
      const Camera& camera = reconstruction_.Camera(camera_id);
      if (!GetAdapter(camera.model_id)) continue;
      GetOrCreateCalibration(camera_id, camera);
    }
  }

  void CreatePoseNodes() {
    // Build a sorted, deduplicated list of (frame_id, model_id) pairs.
    // Each frame has one camera model (single camera per rig assumed).
    std::map<frame_t, CameraModelId> frame_to_model;
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = reconstruction_.Camera(image.CameraId());
      frame_to_model.emplace(image.FrameId(), camera.model_id);
    }
    for (const auto& [frame_id, model_id] : frame_to_model)
      GetOrCreatePose(frame_id, model_id);
  }

  void CreatePointNodes() {
    std::vector<point3D_t> sorted_point_ids;
    for (const auto& [point_id, _] : reconstruction_.Points3D()) {
      if (!config_.IsIgnoredPoint(point_id))
        sorted_point_ids.push_back(point_id);
    }
    std::sort(sorted_point_ids.begin(), sorted_point_ids.end());
    for (const point3D_t point_id : sorted_point_ids)
      GetOrCreatePoint(point_id, reconstruction_.Point3D(point_id));
  }

  void AddFactorsInOptimalOrder() {
    // Iterate calibration-first for optimal GPU memory access patterns.
    // Within each calibration, iterate pose then point.
    for (const auto& [model_id, adapter_ptr] : adapters_) {
      for (size_t calib_idx = 0; calib_idx < calib_num_per_model_.at(model_id);
           ++calib_idx) {
        const camera_t camera_id =
            calib_index_to_camera_.at({model_id, calib_idx});
        AddFactorsForCalibration(camera_id, model_id);
      }
    }
  }

  void AddFactorsForCalibration(camera_t camera_id, CameraModelId model_id) {
    for (const auto& [key, image_ids] : camera_frame_to_images_) {
      if (key.first != camera_id) continue;
      for (const image_t image_id : image_ids) {
        const Image& image = reconstruction_.Image(image_id);
        for (const Point2D& p2d : image.Points2D()) {
          if (!p2d.HasPoint3D() || config_.IsIgnoredPoint(p2d.point3D_id))
            continue;
          if (point_id_to_index_.find(p2d.point3D_id) ==
              point_id_to_index_.end())
            continue;
          AddFactorForObservation(image,
                                  *image.CameraPtr(),
                                  p2d,
                                  reconstruction_.Point3D(p2d.point3D_id));
        }
      }
    }
  }

  void AddExternalFactors() {
    for (const auto point3D_id : config_.VariablePoints())
      AddFactorsForExternalObservations(point3D_id);
    for (const auto point3D_id : config_.ConstantPoints())
      AddFactorsForExternalObservations(point3D_id);
  }

  void AddFactorsForExternalObservations(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));
    Point3D& point3D = reconstruction_.Point3D(point3D_id);
    GetOrCreatePoint(point3D_id, point3D);

    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) continue;
      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      if (!GetAdapter(camera.model_id)) {
        LOG(WARNING) << "Skipping external observation with unsupported "
                        "camera model: "
                     << camera.ModelName();
        continue;
      }
      // Mark frame and camera as external BEFORE AddFactorForObservation so
      // that IsPoseVariable and IsFocalAndExtraVariable correctly return false
      // for all external observations (not just after the first per-camera).
      frames_from_outside_config_.insert(image.FrameId());
      cameras_from_outside_config_.insert(camera.camera_id);
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      AddFactorForObservation(image, camera, point2D, point3D);
    }
  }

  void AddFactorForObservation(const Image& image,
                               const Camera& camera,
                               const Point2D& point2D,
                               const Point3D& point3D) {
    ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
    if (!adapter) return;

    const bool pose_var = IsPoseVariable(image.FrameId());
    const bool focal_and_extra_var = IsFocalAndExtraVariable(camera.camera_id);
    const bool principal_point_var = IsPrincipalPointVariable(camera.camera_id);
    const bool point_var = IsPointVariable(point2D.point3D_id);

    // Detect mismatched refinement flags: CASPAR cannot independently fix
    // focal vs. extra_params within the merged focal_and_extra block.
    if (options_.refine_focal_length != options_.refine_extra_params &&
        !config_.HasConstantCamIntrinsics(camera.camera_id) &&
        !cameras_from_outside_config_.count(camera.camera_id)) {
      LOG_FIRST_N(WARNING, 1)
          << "Camera " << camera.camera_id
          << ": refine_focal_length != refine_extra_params is not supported "
             "by CASPAR's merged focal_and_extra block. Observations skipped.";
      return;
    }

    // For non-ref cameras with a variable rig pose, Caspar can't express
    // project(CamFromRig * RigFromWorld * point, calib) as a variable-pose
    // factor.
    if (!image.IsRefInFrame() && pose_var) return;

    const bool effective_pose_var = pose_var && image.IsRefInFrame();

    // Skip fully-constant observations — nothing to optimize
    if (!effective_pose_var && !focal_and_extra_var && !principal_point_var &&
        !point_var)
      return;

    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);
    ModelData& md = model_data_per_model_.at(camera.model_id);

    // Select the variant from the 4-dimensional (pose, focal_and_extra,
    // principal_point, point) fixed/tunable space.
    FactorVariant v;
    if (effective_pose_var && focal_and_extra_var && principal_point_var &&
        point_var)
      v = FactorVariant::BASE;
    else if (!effective_pose_var && focal_and_extra_var &&
             principal_point_var && point_var)
      v = FactorVariant::FIXED_POSE;
    else if (effective_pose_var && !focal_and_extra_var &&
             principal_point_var && point_var)
      v = FactorVariant::FIXED_FOCAL_AND_EXTRA;
    else if (effective_pose_var && focal_and_extra_var &&
             !principal_point_var && point_var)
      v = FactorVariant::FIXED_PRINCIPAL_POINT;
    else if (effective_pose_var && focal_and_extra_var && principal_point_var &&
             !point_var)
      v = FactorVariant::FIXED_POINT;
    else if (!effective_pose_var && !focal_and_extra_var &&
             principal_point_var && point_var)
      v = FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA;
    else if (!effective_pose_var && focal_and_extra_var &&
             !principal_point_var && point_var)
      v = FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT;
    else if (!effective_pose_var && focal_and_extra_var &&
             principal_point_var && !point_var)
      v = FactorVariant::FIXED_POSE_FIXED_POINT;
    else if (effective_pose_var && !focal_and_extra_var &&
             !principal_point_var && point_var)
      v = FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
    else if (effective_pose_var && !focal_and_extra_var &&
             principal_point_var && !point_var)
      v = FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT;
    else if (effective_pose_var && focal_and_extra_var &&
             !principal_point_var && !point_var)
      v = FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT;
    else if (!effective_pose_var && !focal_and_extra_var &&
             !principal_point_var && point_var)
      v = FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
    else if (!effective_pose_var && !focal_and_extra_var &&
             principal_point_var && !point_var)
      v = FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT;
    else if (!effective_pose_var && focal_and_extra_var &&
             !principal_point_var && !point_var)
      v = FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT;
    else  // pose && !focal_and_extra && !principal_point && !point
      v = FactorVariant::
          FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT;

    VariantData& vd = md.variants[static_cast<int>(v)];

    if (effective_pose_var)
      vd.pose_indices.push_back(GetOrCreatePose(image.FrameId(), camera.model_id));
    else
      // Use CamFromWorld() which correctly computes CamFromRig * RigFromWorld
      // for non-ref cameras, and equals RigFromWorld for ref cameras.
      // Debug test
      AppendPose(vd.const_poses, reconstruction_.Frame(image.FrameId()).RigFromWorld());

    if (focal_and_extra_var) {
      vd.focal_and_extra_indices.push_back(calib_idx);
    } else {
      const size_t fs = adapter->FocalAndExtraSize();
      for (size_t i = 0; i < fs; ++i)
        vd.const_focal_and_extra.push_back(
            md.focal_and_extra_data[calib_idx * fs + i]);
    }

    if (principal_point_var) {
      vd.principal_point_indices.push_back(calib_idx);
    } else {
      const size_t ps = adapter->PrincipalPointSize();
      for (size_t i = 0; i < ps; ++i)
        vd.const_principal_point.push_back(
            md.principal_point_data[calib_idx * ps + i]);
    }

    if (point_var)
      vd.point_indices.push_back(GetOrCreatePoint(point2D.point3D_id, point3D));
    else
      AppendPoint(vd.const_points, point3D);

    vd.pixels.push_back(point2D.xy.x());
    vd.pixels.push_back(point2D.xy.y());
    ++vd.num_factors;
  }

  static void AppendPose(std::vector<StorageType>& out, const Rigid3d& pose) {
    out.push_back(pose.rotation().x());
    out.push_back(pose.rotation().y());
    out.push_back(pose.rotation().z());
    out.push_back(pose.rotation().w());
    out.push_back(pose.translation().x());
    out.push_back(pose.translation().y());
    out.push_back(pose.translation().z());
  }

  static void AppendPoint(std::vector<StorageType>& out, const Point3D& pt) {
    out.push_back(pt.xyz.x());
    out.push_back(pt.xyz.y());
    out.push_back(pt.xyz.z());
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

  size_t GetOrCreatePose(const frame_t frame_id, const CameraModelId model_id) {
    size_t& n = num_poses_per_model_[model_id];
    auto [it, inserted] =
        frame_to_pose_index_per_model_[model_id].try_emplace(frame_id, n);
    if (inserted) {
      pose_index_to_frame_per_model_[model_id][n] = frame_id;
      const Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      auto& data = pose_data_per_model_[model_id];
      data.push_back(pose.rotation().x());
      data.push_back(pose.rotation().y());
      data.push_back(pose.rotation().z());
      data.push_back(pose.rotation().w());
      data.push_back(pose.translation().x());
      data.push_back(pose.translation().y());
      data.push_back(pose.translation().z());
      n++;
    }
    return it->second;
  }

  // Calib indices are per-model: index 0 for SimpleRadial is unrelated to
  // index 0 for Pinhole. The key is (model_id, calib_idx). The same index
  // addresses both focal_data and extra_calib_data for that model.
  size_t GetOrCreateCalibration(const camera_t camera_id,
                                const Camera& camera) {
    auto [it, inserted] = camera_to_calib_index_.try_emplace(camera_id, 0);
    if (inserted) {
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      size_t& model_calib_count = calib_num_per_model_[camera.model_id];
      it->second = model_calib_count;
      calib_index_to_camera_[{camera.model_id, model_calib_count}] = camera_id;
      ModelData& md = model_data_per_model_.at(camera.model_id);
      adapter->ExtractFocalAndExtra(camera, md.focal_and_extra_data);
      adapter->ExtractPrincipalPoint(camera, md.principal_point_data);
      ++model_calib_count;
    }
    return it->second;
  }

  bool IsPoseVariable(const frame_t frame_id) const {
    if (!options_.refine_rig_from_world) return false;
    if (config_.HasConstantRigFromWorldPose(frame_id)) return false;
    if (frames_from_outside_config_.count(frame_id)) return false;
    return true;
  }

  // Both focal and extra_params must be refined together (merged block).
  // If they disagree, observations are skipped — see AddFactorForObservation.
  bool IsFocalAndExtraVariable(const camera_t camera_id) const {
    if (!options_.refine_focal_length || !options_.refine_extra_params)
      return false;
    if (config_.HasConstantCamIntrinsics(camera_id)) return false;
    if (cameras_from_outside_config_.count(camera_id)) return false;
    return true;
  }

  bool IsPrincipalPointVariable(const camera_t camera_id) const {
    if (!options_.refine_principal_point) return false;
    if (config_.HasConstantCamIntrinsics(camera_id)) return false;
    if (cameras_from_outside_config_.count(camera_id)) return false;
    return true;
  }

  bool AreIntrinsicsVariable(const camera_t camera_id) const {
    return IsFocalAndExtraVariable(camera_id) ||
           IsPrincipalPointVariable(camera_id);
  }

  bool IsPointVariable(const point3D_t point3D_id) const {
    if (config_.HasConstantPoint(point3D_id)) return false;
    auto it = point3D_num_observations_.find(point3D_id);
    if (it == point3D_num_observations_.end()) return false;
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    if (point3D.track.Length() > it->second) return false;
    return true;
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    VLOG(2) << "=== CASPAR SOLVER SETUP ===";
    VLOG(2) << "  Points: " << num_points_;

    if (num_points_ > 0)
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      const ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);
      const size_t n_poses = num_poses_per_model_.count(model_id)
                                 ? num_poses_per_model_.at(model_id)
                                 : 0;

      VLOG(2) << "  Poses (" << static_cast<int>(model_id) << "): " << n_poses;

      if (n_poses > 0)
        adapter_ptr->SetPoseNodes(
            solver, pose_data_per_model_.at(model_id).data(), n_poses);

      if (n_calib > 0) {
        adapter_ptr->SetFocalAndExtraNodes(
            solver,
            const_cast<StorageType*>(md.focal_and_extra_data.data()),
            n_calib);
        adapter_ptr->SetPrincipalPointNodes(
            solver,
            const_cast<StorageType*>(md.principal_point_data.data()),
            n_calib);

        // Set merged Calib nodes when both intrinsic groups are tunable.
        const bool has_merged =
            md.variants[static_cast<int>(FactorVariant::BASE)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POINT)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE_FIXED_POINT)]
                    .num_factors > 0;
        if (has_merged) {
          const size_t fae_size = adapter_ptr->FocalAndExtraSize();
          const size_t pp_size  = adapter_ptr->PrincipalPointSize();
          const size_t cal_size = adapter_ptr->CalibSize();
          std::vector<StorageType> calib_data(n_calib * cal_size);
          for (size_t i = 0; i < n_calib; ++i) {
            for (size_t j = 0; j < fae_size; ++j)
              calib_data[i * cal_size + j] =
                  md.focal_and_extra_data[i * fae_size + j];
            for (size_t j = 0; j < pp_size; ++j)
              calib_data[i * cal_size + fae_size + j] =
                  md.principal_point_data[i * pp_size + j];
          }
          if (n_calib > 0) {
            VLOG(2) << "  SetCalibNodes [cam 0, model "
                    << static_cast<int>(model_id) << "]: ["
                    << calib_data[0] << ", " << calib_data[1] << ", "
                    << calib_data[2] << ", " << calib_data[3] << "]";
          }
          adapter_ptr->SetCalibNodes(solver, calib_data.data(), n_calib);
        }
      }
      for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
        if (md.variants[v].num_factors > 0)
          adapter_ptr->SetVariantFactors(
              solver, static_cast<FactorVariant>(v), md.variants[v]);
      }
    }
    solver.finish_indices();
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0)
      solver.get_Point_nodes_to_stacked_host(
          point_data_.data(), 0, num_points_);

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      const size_t n_poses = num_poses_per_model_.count(model_id)
                                 ? num_poses_per_model_.at(model_id)
                                 : 0;
      if (n_poses > 0)
        adapter_ptr->GetPoseNodes(
            solver, pose_data_per_model_.at(model_id).data(), n_poses);
    }

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);
      if (n_calib > 0) {
        adapter_ptr->GetFocalAndExtraNodes(
            solver, md.focal_and_extra_data.data(), n_calib);
        adapter_ptr->GetPrincipalPointNodes(
            solver, md.principal_point_data.data(), n_calib);

        // If merged variants were active, split the merged Calib node back into
        // focal_and_extra_data and principal_point_data (overwrites the stale
        // split-pool values read above).
        const bool has_merged =
            md.variants[static_cast<int>(FactorVariant::BASE)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POINT)].num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE_FIXED_POINT)]
                    .num_factors > 0;
        if (has_merged) {
          const size_t fae_size = adapter_ptr->FocalAndExtraSize();
          const size_t pp_size  = adapter_ptr->PrincipalPointSize();
          const size_t cal_size = adapter_ptr->CalibSize();
          std::vector<StorageType> calib_data(n_calib * cal_size);
          adapter_ptr->GetCalibNodes(solver, calib_data.data(), n_calib);
          if (n_calib > 0) {
            VLOG(2) << "  GetCalibNodes [cam 0, model "
                    << static_cast<int>(model_id) << "]: ["
                    << calib_data[0] << ", " << calib_data[1] << ", "
                    << calib_data[2] << ", " << calib_data[3] << "]";
          }
          for (size_t i = 0; i < n_calib; ++i) {
            for (size_t j = 0; j < fae_size; ++j)
              md.focal_and_extra_data[i * fae_size + j] =
                  calib_data[i * cal_size + j];
            for (size_t j = 0; j < pp_size; ++j)
              md.principal_point_data[i * pp_size + j] =
                  calib_data[i * cal_size + fae_size + j];
          }
          if (n_calib > 0) {
            VLOG(2) << "  After split-back [cam 0]: fae=["
                    << md.focal_and_extra_data[0] << ", "
                    << md.focal_and_extra_data[1] << "] pp=["
                    << md.principal_point_data[0] << ", "
                    << md.principal_point_data[1] << "]";
          }
        }
      }
    }
  }

  void WriteCalibsToReconstruction() {
    for (const auto& [camera_id, calib_idx] : camera_to_calib_index_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;
      Camera& camera = reconstruction_.Camera(camera_id);
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      const ModelData& md = model_data_per_model_.at(camera.model_id);
      if (IsFocalAndExtraVariable(camera_id))
        adapter->WriteFocalAndExtra(
            camera, md.focal_and_extra_data.data(), calib_idx);
      if (IsPrincipalPointVariable(camera_id))
        adapter->WritePrincipalPoint(
            camera, md.principal_point_data.data(), calib_idx);
      THROW_CHECK(camera.VerifyParams());
    }
  }

  void WriteResultsToReconstruction() {
    for (const auto& [idx, point_id] : index_to_point_id_) {
      if (config_.HasConstantPoint(point_id)) continue;
      // Skip points that are not truly variable (e.g. partially contained
      // tracks with external observations). Their solver nodes hold float
      // copies of the original double values and must not be written back.
      if (!IsPointVariable(point_id)) continue;
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }

    for (const auto& [model_id, idx_to_frame] : pose_index_to_frame_per_model_) {
      const auto& data = pose_data_per_model_.at(model_id);
      for (const auto& [idx, frame_id] : idx_to_frame) {
        if (!IsPoseVariable(frame_id)) continue;
        Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
        pose.rotation().x() = data[idx * 7 + 0];
        pose.rotation().y() = data[idx * 7 + 1];
        pose.rotation().z() = data[idx * 7 + 2];
        pose.rotation().w() = data[idx * 7 + 3];
        pose.translation().x() = data[idx * 7 + 4];
        pose.translation().y() = data[idx * 7 + 5];
        pose.translation().z() = data[idx * 7 + 6];
        pose.rotation().normalize();
      }
    }

    WriteCalibsToReconstruction();
  }

  size_t ComputeTotalResiduals() const {
    size_t total = 0;
    for (const auto& [model_id, md] : model_data_per_model_) {
      for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
        total += md.variants[v].num_factors;
      }
    }
    return 2 * total;
  }

  CasparSolverSizing BuildSizing() const {
    CasparSolverSizing sz;
    sz.num_points = num_points_;
    if (auto it = num_poses_per_model_.find(CameraModelId::kSimpleRadial);
        it != num_poses_per_model_.end())
      sz.num_simple_radial_poses = it->second;
    if (auto it = num_poses_per_model_.find(CameraModelId::kPinhole);
        it != num_poses_per_model_.end())
      sz.num_pinhole_poses = it->second;

    auto get_md = [&](CameraModelId id) -> const ModelData* {
      auto it = model_data_per_model_.find(id);
      return it != model_data_per_model_.end() ? &it->second : nullptr;
    };
    auto get_n = [&](CameraModelId id) -> size_t {
      auto it = calib_num_per_model_.find(id);
      return it != calib_num_per_model_.end() ? it->second : 0;
    };

    if (const ModelData* md = get_md(CameraModelId::kSimpleRadial)) {
      sz.num_simple_radial_calibs = get_n(CameraModelId::kSimpleRadial);
      adapters_.at(CameraModelId::kSimpleRadial)
          ->FillSizing(sz, *md, sz.num_simple_radial_calibs);
    }
    if (const ModelData* md = get_md(CameraModelId::kPinhole)) {
      sz.num_pinhole_calibs = get_n(CameraModelId::kPinhole);
      adapters_.at(CameraModelId::kPinhole)
          ->FillSizing(sz, *md, sz.num_pinhole_calibs);
    }
    return sz;
  }

  size_t TotalPoses() const {
    size_t n = 0;
    for (const auto& [_, count] : num_poses_per_model_) n += count;
    return n;
  }

  bool ValidateData() const {
    if (num_points_ == 0 && TotalPoses() == 0) {
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

    caspar::SolverParams<StorageType> params;

    params.pcg_iter_max = 50;
    params.solver_iter_max = 400;
    params.diag_min = 1e-6;
    params.solver_rel_decrease_min = 0.999;

    auto solver = CreateSolver(params, BuildSizing());
    SetupSolverData(solver);
    caspar::SolveResult result = solver.solve(/*print_progress=*/false);
    ReadSolverResults(solver);
    WriteResultsToReconstruction();

    auto summary = CasparBundleAdjustmentSummary::Create(result);
    summary->num_residuals = ComputeTotalResiduals();
    return summary;
  }

  // --- Per-model state ---
  std::unordered_map<CameraModelId, std::unique_ptr<ICasparModelAdapter>>
      adapters_;
  std::unordered_map<CameraModelId, ModelData> model_data_per_model_;
  std::unordered_map<CameraModelId, size_t> calib_num_per_model_;

  // camera_id -> calib index within its model's array
  std::unordered_map<camera_t, size_t> camera_to_calib_index_;
  // (model_id, calib_idx) -> camera_id  (for write-back)
  std::map<std::pair<CameraModelId, size_t>, camera_t> calib_index_to_camera_;

  Reconstruction& reconstruction_;
  size_t num_points_ = 0;
  std::vector<StorageType> point_data_;

  // Pose pools are per-model so that SimpleRadial and Pinhole factors are
  // never batched into the same Caspar block (reducing shared memory use).
  std::unordered_map<CameraModelId, size_t> num_poses_per_model_;
  std::unordered_map<CameraModelId, std::vector<StorageType>> pose_data_per_model_;
  std::unordered_map<CameraModelId, std::unordered_map<frame_t, size_t>>
      frame_to_pose_index_per_model_;
  std::unordered_map<CameraModelId, std::unordered_map<size_t, frame_t>>
      pose_index_to_frame_per_model_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;

  std::unordered_set<frame_t> frames_from_outside_config_;
  std::unordered_set<camera_t> cameras_from_outside_config_;
  std::unordered_map<std::pair<camera_t, frame_t>,
                     std::vector<image_t>,
                     PairHash<camera_t, frame_t>>
      camera_frame_to_images_;

  std::unordered_map<point3D_t, size_t> point3D_num_observations_;
};
}  // namespace

std::shared_ptr<CasparBundleAdjustmentSummary>
CasparBundleAdjustmentSummary::Create(
    const caspar::SolveResult& caspar_summary) {
  auto summary = std::make_shared<CasparBundleAdjustmentSummary>();
  switch (caspar_summary.exit_reason) {
    case caspar::ExitReason::CONVERGED_DIAG_EXIT:
      VLOG(1) << "Caspar: CONVERGED_DIAG_EXIT after "
              << caspar_summary.iteration_count << " iters"
              << " (diag limit hit — likely premature termination)";
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case caspar::ExitReason::CONVERGED_SCORE_THRESHOLD:
      VLOG(1) << "Caspar: CONVERGED_SCORE_THRESHOLD after "
              << caspar_summary.iteration_count << " iters";
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case caspar::ExitReason::MAX_ITERATIONS:
      VLOG(1) << "Caspar: MAX_ITERATIONS (" << caspar_summary.iteration_count
              << ")";
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