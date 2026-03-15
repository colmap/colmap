#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/camera.h"
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
    VLOG(1) << "Using Caspar bu ndle adjuster";

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
    std::vector<frame_t> sorted_frame_ids;
    for (const image_t image_id : config_.Images())
      sorted_frame_ids.push_back(reconstruction_.Image(image_id).FrameId());
    std::sort(sorted_frame_ids.begin(), sorted_frame_ids.end());
    sorted_frame_ids.erase(
        std::unique(sorted_frame_ids.begin(), sorted_frame_ids.end()),
        sorted_frame_ids.end());

    for (const frame_t frame_id : sorted_frame_ids) GetOrCreatePose(frame_id);
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
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      AddFactorForObservation(image, camera, point2D, point3D);
      cameras_from_outside_config_.insert(camera.camera_id);
    }
  }

  void AddFactorForObservation(const Image& image,
                               const Camera& camera,
                               const Point2D& point2D,
                               const Point3D& point3D) {
    ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
    if (!adapter) return;

    const bool pose_var = IsPoseVariable(image.FrameId());
    const bool calib_var = AreIntrinsicsVariable(camera.camera_id);
    const bool point_var = IsPointVariable(point2D.point3D_id);

    // Skip fully-constant observations — nothing to optimize
    if (!pose_var && !calib_var && !point_var) return;

    // Fixed-calib variants are not supported: skip observations where
    // calibration would be constant but pose or point are variable.
    if (!calib_var) {
      LOG(WARNING) << "Skipping observation: fixed-calib variants not "
                      "supported. "
                   << camera.camera_id;
      return;
    }

    const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);
    ModelData& md = model_data_per_model_.at(camera.model_id);

    FactorVariant v;
    if (pose_var && point_var)
      v = FactorVariant::BASE;
    else if (!pose_var && point_var)
      v = FactorVariant::FIXED_POSE;
    else if (pose_var && !point_var)
      v = FactorVariant::FIXED_POINT;
    else
      v = FactorVariant::FIXED_POSE_FIXED_POINT;

    VariantData& vd = md.variants[static_cast<int>(v)];
    vd.calib_indices.push_back(calib_idx);

    if (pose_var)
      vd.pose_indices.push_back(GetOrCreatePose(image.FrameId()));
    else
      AppendPose(vd.const_poses, image.FramePtr()->RigFromWorld());

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

  // Calib indices are per-model: index 0 for SimpleRadial is unrelated to
  // index 0 for Pinhole. The key is (model_id, calib_idx).
  size_t GetOrCreateCalibration(const camera_t camera_id,
                                const Camera& camera) {
    auto [it, inserted] = camera_to_calib_index_.try_emplace(camera_id, 0);
    if (inserted) {
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      size_t& model_calib_count = calib_num_per_model_[camera.model_id];
      it->second = model_calib_count;
      calib_index_to_camera_[{camera.model_id, model_calib_count}] = camera_id;
      adapter->ExtractCalib(
          camera, model_data_per_model_.at(camera.model_id).calib_data);
      ++model_calib_count;
    }
    return it->second;
  }

  bool IsPoseVariable(const frame_t frame_id) const {
    if (!options_.refine_rig_from_world) return false;
    if (config_.HasConstantRigFromWorldPose(frame_id)) return false;
    return true;
  }

  bool AreIntrinsicsVariable(const camera_t camera_id) const {
    const bool any_refinement = options_.refine_focal_length ||
                                options_.refine_principal_point ||
                                options_.refine_extra_params;
    if (!any_refinement) return false;
    if (config_.HasConstantCamIntrinsics(camera_id)) return false;
    if (cameras_from_outside_config_.count(camera_id)) return false;
    return true;
  }

  bool IsPointVariable(const point3D_t point3D_id) const {
    if (config_.HasConstantPoint(point3D_id)) return false;
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    if (point3D.track.Length() > point3D_num_observations_.at(point3D_id))
      return false;
    return true;
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    VLOG(2) << "=== CASPAR SOLVER SETUP ===";
    VLOG(2) << "  Points: " << num_points_;
    VLOG(2) << "  Poses:  " << num_poses_;

    if (num_points_ > 0)
      solver.set_Point_nodes_from_stacked_host(
          point_data_.data(), 0, num_points_);
    if (num_poses_ > 0)
      solver.set_Pose_nodes_from_stacked_host(pose_data_.data(), 0, num_poses_);

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      const ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);

      if (n_calib > 0)
        adapter_ptr->SetCalibNodes(
            solver, const_cast<StorageType*>(md.calib_data.data()), n_calib);
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
    if (num_poses_ > 0)
      solver.get_Pose_nodes_to_stacked_host(pose_data_.data(), 0, num_poses_);

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);
      if (n_calib > 0)
        adapter_ptr->GetCalibNodes(solver, md.calib_data.data(), n_calib);
    }
  }

  void WriteCalibsToReconstruction() {
    for (const auto& [camera_id, calib_idx] : camera_to_calib_index_) {
      if (!AreIntrinsicsVariable(camera_id)) continue;
      Camera& camera = reconstruction_.Camera(camera_id);
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      const ModelData& md = model_data_per_model_.at(camera.model_id);
      adapter->WriteCalib(camera, md.calib_data.data(), calib_idx);
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
    sz.num_poses = num_poses_;
    sz.num_points = num_points_;

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

  bool ValidateData() const {
    if (num_points_ == 0 && num_poses_ == 0) {
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
    auto solver = CreateSolver(params, BuildSizing());
    SetupSolverData(solver);
    caspar::SolveResult result = solver.solve(/*print_progress=*/true);
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
  size_t num_poses_ = 0;
  std::vector<StorageType> point_data_;
  std::vector<StorageType> pose_data_;

  std::unordered_map<point3D_t, size_t> point_id_to_index_;
  std::unordered_map<size_t, point3D_t> index_to_point_id_;
  std::unordered_map<frame_t, size_t> frame_to_pose_index_;
  std::unordered_map<size_t, frame_t> pose_index_to_frame_;

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
    case caspar::ExitReason::CONVERGED_SCORE_THRESHOLD:
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case caspar::ExitReason::MAX_ITERATIONS:
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