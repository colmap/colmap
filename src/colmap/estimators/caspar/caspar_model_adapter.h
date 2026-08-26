#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"

#include <memory>
#include <vector>

#ifdef CASPAR_USE_DOUBLE
#include "thirdparty/Symforce-Caspar/generated/f64/solver.h"
#else
#include "thirdparty/Symforce-Caspar/generated/f32/solver.h"
#endif

namespace colmap {

struct CasparSolverSizing {
  // Pose pools are per-model to prevent cross-model factor batching.
  size_t num_simple_radial_poses = 0;
  size_t num_pinhole_poses = 0;
  size_t num_thin_prism_fisheye_poses = 0;
  size_t num_points = 0;

  // SimpleRadial: num_calibs is shared by the merged Calib pool and the split
  // FocalAndExtra / PrincipalPoint pools, one entry per camera.
  // Merged variants (both intrinsic groups tunable): 4 counts below.
  // Split variants (at least one group fixed): 11 counts below.
  size_t num_simple_radial_calibs = 0;
  size_t num_simple_radial = 0;
  size_t num_simple_radial_fixed_pose = 0;
  size_t num_simple_radial_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_point = 0;
  size_t num_simple_radial_split_fixed_focal_and_extra = 0;
  size_t num_simple_radial_split_fixed_principal_point = 0;
  size_t num_simple_radial_split_fixed_pose_fixed_focal_and_extra = 0;
  size_t num_simple_radial_split_fixed_pose_fixed_principal_point = 0;
  size_t num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point =
      0;
  size_t num_simple_radial_split_fixed_focal_and_extra_fixed_point = 0;
  size_t num_simple_radial_split_fixed_principal_point_fixed_point = 0;
  size_t
      num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point =
          0;
  size_t num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point =
      0;
  size_t num_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point =
      0;
  size_t
      num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point =
          0;

  // Pinhole: same layout as SimpleRadial above.
  size_t num_pinhole_calibs = 0;
  size_t num_pinhole = 0;
  size_t num_pinhole_fixed_pose = 0;
  size_t num_pinhole_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_point = 0;
  size_t num_pinhole_split_fixed_focal = 0;
  size_t num_pinhole_split_fixed_principal_point = 0;
  size_t num_pinhole_split_fixed_pose_fixed_focal = 0;
  size_t num_pinhole_split_fixed_pose_fixed_principal_point = 0;
  size_t num_pinhole_split_fixed_focal_fixed_principal_point = 0;
  size_t num_pinhole_split_fixed_focal_fixed_point = 0;
  size_t num_pinhole_split_fixed_principal_point_fixed_point = 0;
  size_t num_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point = 0;
  size_t num_pinhole_split_fixed_pose_fixed_focal_fixed_point = 0;
  size_t num_pinhole_split_fixed_pose_fixed_principal_point_fixed_point = 0;
  size_t num_pinhole_split_fixed_focal_fixed_principal_point_fixed_point = 0;

  // ThinPrismFisheye: same 4 merged + 11 split variant layout.
  size_t num_thin_prism_fisheye_calibs = 0;
  size_t num_thin_prism_fisheye = 0;
  size_t num_thin_prism_fisheye_fixed_pose = 0;
  size_t num_thin_prism_fisheye_fixed_point = 0;
  size_t num_thin_prism_fisheye_fixed_pose_fixed_point = 0;
  size_t num_thin_prism_fisheye_split_fixed_focal_and_extra = 0;
  size_t num_thin_prism_fisheye_split_fixed_principal_point = 0;
  size_t num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra = 0;
  size_t num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point = 0;
  size_t
      num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point =
          0;
  size_t num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point = 0;
  size_t num_thin_prism_fisheye_split_fixed_principal_point_fixed_point = 0;
  size_t
      num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point =
          0;
  size_t
      num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point =
          0;
  size_t
      num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point =
          0;
  size_t
      num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point =
          0;
};

// One implementation per camera model.

class ICasparModelAdapter {
 public:
  virtual ~ICasparModelAdapter() = default;

  virtual CameraModelId ModelId() const = 0;

  // Number of floats in the focal_and_extra/focal and principal_point
  // node arrays per camera.
  virtual size_t FocalAndExtraSize() const = 0;
  virtual size_t PrincipalPointSize() const = 0;
  // Number of floats in the merged Calib node per camera
  // (= FocalAndExtraSize() + PrincipalPointSize()).
  virtual size_t CalibSize() const = 0;

  virtual void MergeCalibData(const StorageType* focal_and_extra,
                              const StorageType* principal_point,
                              StorageType* calib) const {
    const size_t fae_size = FocalAndExtraSize();
    const size_t pp_size = PrincipalPointSize();
    for (size_t i = 0; i < fae_size; ++i) {
      calib[i] = focal_and_extra[i];
    }
    for (size_t i = 0; i < pp_size; ++i) {
      calib[fae_size + i] = principal_point[i];
    }
  }

  virtual void SplitCalibData(const StorageType* calib,
                              StorageType* focal_and_extra,
                              StorageType* principal_point) const {
    const size_t fae_size = FocalAndExtraSize();
    const size_t pp_size = PrincipalPointSize();
    for (size_t i = 0; i < fae_size; ++i) {
      focal_and_extra[i] = calib[i];
    }
    for (size_t i = 0; i < pp_size; ++i) {
      principal_point[i] = calib[fae_size + i];
    }
  }

  virtual void SetCalibNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;
  virtual void GetCalibNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;

  virtual void FillSizing(CasparSolverSizing& sz,
                          const ModelData& md,
                          size_t num_calibs) const = 0;

  // Append focal_and_extra / focal / principal_point params from a
  // camera into a flat output vector.
  virtual void ExtractFocalAndExtra(const Camera& camera,
                                    std::vector<StorageType>& out) const = 0;
  virtual void ExtractPrincipalPoint(const Camera& camera,
                                     std::vector<StorageType>& out) const = 0;

  virtual void WriteFocalAndExtra(Camera& camera,
                                  const StorageType* focal_and_extra_data,
                                  size_t idx) const = 0;
  virtual void WritePrincipalPoint(Camera& camera,
                                   const StorageType* principal_point_data,
                                   size_t idx) const = 0;

  virtual void SetPoseNodes(caspar::GraphSolver& solver,
                            StorageType* data,
                            size_t n) const = 0;
  virtual void GetPoseNodes(caspar::GraphSolver& solver,
                            StorageType* data,
                            size_t n) const = 0;

  virtual void SetFocalAndExtraNodes(caspar::GraphSolver& solver,
                                     StorageType* data,
                                     size_t n) const = 0;
  virtual void GetFocalAndExtraNodes(caspar::GraphSolver& solver,
                                     StorageType* data,
                                     size_t n) const = 0;
  virtual void SetPrincipalPointNodes(caspar::GraphSolver& solver,
                                      StorageType* data,
                                      size_t n) const = 0;
  virtual void GetPrincipalPointNodes(caspar::GraphSolver& solver,
                                      StorageType* data,
                                      size_t n) const = 0;

  virtual void SetVariantFactors(caspar::GraphSolver& solver,
                                 FactorVariant variant,
                                 const VariantData& data) const = 0;
};

// SimpleRadial implementation

class SimpleRadialAdapter : public ICasparModelAdapter {
 public:
  CameraModelId ModelId() const override {
    return CameraModelId::kSimpleRadial;
  }
  // SimpleRadial: params = [f, cx, cy, k]
  // focal_and_extra = [f, k]  (non-contiguous in params array)
  // principal_point      = [cx, cy]
  // merged calib         = [f, k, cx, cy]
  size_t FocalAndExtraSize() const override { return 2; }
  size_t PrincipalPointSize() const override { return 2; }
  size_t CalibSize() const override {
    return FocalAndExtraSize() + PrincipalPointSize();
  }

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_simple_radial_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        // Merged variants: both focal_and_extra and principal_point are
        // tunable.
        case FactorVariant::BASE:
          sz.num_simple_radial = n;
          break;
        case FactorVariant::FIXED_POSE:
          sz.num_simple_radial_fixed_pose = n;
          break;
        case FactorVariant::FIXED_POINT:
          sz.num_simple_radial_fixed_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_point = n;
          break;
        // Split variants: at least one intrinsic group is fixed.
        case FactorVariant::FIXED_FOCAL_AND_EXTRA:
          sz.num_simple_radial_split_fixed_focal_and_extra = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_split_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
          sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_split_fixed_pose_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point =
              n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_simple_radial_split_fixed_focal_and_extra_fixed_point = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_split_fixed_principal_point_fixed_point = n;
          break;
        case FactorVariant::
            FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point =
              n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point =
              n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point =
              n;
          break;
        case FactorVariant::
            FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point =
              n;
          break;
      }
    }
  }

  void ExtractFocalAndExtra(const Camera& camera,
                            std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[0]));  // f
    out.push_back(static_cast<StorageType>(camera.params[3]));  // k
  }

  void ExtractPrincipalPoint(const Camera& camera,
                             std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[1]));  // cx
    out.push_back(static_cast<StorageType>(camera.params[2]));  // cy
  }

  void WriteFocalAndExtra(Camera& camera,
                          const StorageType* data,
                          size_t idx) const override {
    camera.params[0] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 0]);  // f
    camera.params[3] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 1]);  // k
  }

  void WritePrincipalPoint(Camera& camera,
                           const StorageType* data,
                           size_t idx) const override {
    camera.params[1] =
        static_cast<double>(data[idx * PrincipalPointSize() + 0]);  // cx
    camera.params[2] =
        static_cast<double>(data[idx * PrincipalPointSize() + 1]);  // cy
  }

  void SetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.SetSimpleRadialPoseNodesFromStackedHost(data, 0, n);
  }

  void GetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.GetSimpleRadialPoseNodesToStackedHost(data, 0, n);
  }

  void SetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.SetSimpleRadialFocalAndExtraNodesFromStackedHost(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.GetSimpleRadialFocalAndExtraNodesToStackedHost(data, 0, n);
  }

  void SetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.SetSimpleRadialPrincipalPointNodesFromStackedHost(data, 0, n);
  }

  void GetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.GetSimpleRadialPrincipalPointNodesToStackedHost(data, 0, n);
  }

  void SetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.SetSimpleRadialCalibNodesFromStackedHost(data, 0, n);
  }

  void GetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.GetSimpleRadialCalibNodesToStackedHost(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      // Merged variants: the calib index is the same as
      // focal_and_extra_index, so no VariantData changes are needed for
      // these cases.
      case FactorVariant::BASE:
        s.SetSimpleRadialNum(n);
        s.SetSimpleRadialPoseIndicesFromHost(d.pose_indices.data(), n);
        s.SetSimpleRadialSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialCalibIndicesFromHost(d.focal_and_extra_indices.data(),
                                              n);
        s.SetSimpleRadialPointIndicesFromHost(d.point_indices.data(), n);
        s.SetSimpleRadialPixelDataFromStackedHost(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.SetSimpleRadialFixedPoseNum(n);
        s.SetSimpleRadialFixedPoseSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialFixedPoseCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPosePointIndicesFromHost(d.point_indices.data(),
                                                       n);
        s.SetSimpleRadialFixedPosePoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPosePixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.SetSimpleRadialSplitFixedFocalAndExtraNum(n);
        s.SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialSplitFixedPrincipalPointNum(n);
        s.SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.SetSimpleRadialFixedPointNum(n);
        s.SetSimpleRadialFixedPointPoseIndicesFromHost(d.pose_indices.data(),
                                                       n);
        s.SetSimpleRadialFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraNum(n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointNum(n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.SetSimpleRadialFixedPoseFixedPointNum(n);
        s.SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointNum(n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointNum(n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
            n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointNum(n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointNum(n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
            n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
    }
  }
};

// ThinPrismFisheye implementation

class ThinPrismFisheyeAdapter : public ICasparModelAdapter {
 public:
  CameraModelId ModelId() const override {
    return CameraModelId::kThinPrismFisheye;
  }
  // ThinPrismFisheye: params =
  // [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]
  // focal_and_extra = [fx, fy, k1, k2, p1, p2, k3, k4, sx1, sy1]
  // principal_point = [cx, cy]
  // merged calib    = COLMAP order above
  size_t FocalAndExtraSize() const override { return 10; }
  size_t PrincipalPointSize() const override { return 2; }
  size_t CalibSize() const override {
    return FocalAndExtraSize() + PrincipalPointSize();
  }

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_thin_prism_fisheye_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        // Merged variants: both focal_and_extra and principal_point are
        // tunable.
        case FactorVariant::BASE:
          sz.num_thin_prism_fisheye = n;
          break;
        case FactorVariant::FIXED_POSE:
          sz.num_thin_prism_fisheye_fixed_pose = n;
          break;
        case FactorVariant::FIXED_POINT:
          sz.num_thin_prism_fisheye_fixed_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_thin_prism_fisheye_fixed_pose_fixed_point = n;
          break;
        // Split variants: at least one intrinsic group is fixed.
        case FactorVariant::FIXED_FOCAL_AND_EXTRA:
          sz.num_thin_prism_fisheye_split_fixed_focal_and_extra = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT:
          sz.num_thin_prism_fisheye_split_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
          sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
          sz.num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point =
              n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_thin_prism_fisheye_split_fixed_principal_point_fixed_point = n;
          break;
        case FactorVariant::
            FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point =
              n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point =
              n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point =
              n;
          break;
        case FactorVariant::
            FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point =
              n;
          break;
      }
    }
  }

  void ExtractFocalAndExtra(const Camera& camera,
                            std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[0]));   // fx
    out.push_back(static_cast<StorageType>(camera.params[1]));   // fy
    out.push_back(static_cast<StorageType>(camera.params[4]));   // k1
    out.push_back(static_cast<StorageType>(camera.params[5]));   // k2
    out.push_back(static_cast<StorageType>(camera.params[6]));   // p1
    out.push_back(static_cast<StorageType>(camera.params[7]));   // p2
    out.push_back(static_cast<StorageType>(camera.params[8]));   // k3
    out.push_back(static_cast<StorageType>(camera.params[9]));   // k4
    out.push_back(static_cast<StorageType>(camera.params[10]));  // sx1
    out.push_back(static_cast<StorageType>(camera.params[11]));  // sy1
  }

  void ExtractPrincipalPoint(const Camera& camera,
                             std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[2]));  // cx
    out.push_back(static_cast<StorageType>(camera.params[3]));  // cy
  }

  void WriteFocalAndExtra(Camera& camera,
                          const StorageType* data,
                          size_t idx) const override {
    camera.params[0] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 0]);  // fx
    camera.params[1] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 1]);  // fy
    camera.params[4] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 2]);  // k1
    camera.params[5] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 3]);  // k2
    camera.params[6] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 4]);  // p1
    camera.params[7] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 5]);  // p2
    camera.params[8] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 6]);  // k3
    camera.params[9] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 7]);  // k4
    camera.params[10] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 8]);  // sx1
    camera.params[11] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 9]);  // sy1
  }

  void WritePrincipalPoint(Camera& camera,
                           const StorageType* data,
                           size_t idx) const override {
    camera.params[2] =
        static_cast<double>(data[idx * PrincipalPointSize() + 0]);  // cx
    camera.params[3] =
        static_cast<double>(data[idx * PrincipalPointSize() + 1]);  // cy
  }

  void MergeCalibData(const StorageType* focal_and_extra,
                      const StorageType* principal_point,
                      StorageType* calib) const override {
    calib[0] = focal_and_extra[0];   // fx
    calib[1] = focal_and_extra[1];   // fy
    calib[2] = principal_point[0];   // cx
    calib[3] = principal_point[1];   // cy
    calib[4] = focal_and_extra[2];   // k1
    calib[5] = focal_and_extra[3];   // k2
    calib[6] = focal_and_extra[4];   // p1
    calib[7] = focal_and_extra[5];   // p2
    calib[8] = focal_and_extra[6];   // k3
    calib[9] = focal_and_extra[7];   // k4
    calib[10] = focal_and_extra[8];  // sx1
    calib[11] = focal_and_extra[9];  // sy1
  }

  void SplitCalibData(const StorageType* calib,
                      StorageType* focal_and_extra,
                      StorageType* principal_point) const override {
    focal_and_extra[0] = calib[0];  // fx
    focal_and_extra[1] = calib[1];  // fy
    principal_point[0] = calib[2];  // cx
    principal_point[1] = calib[3];  // cy
    focal_and_extra[2] = calib[4];  // k1
    focal_and_extra[3] = calib[5];  // k2
    focal_and_extra[4] = calib[6];  // p1
    focal_and_extra[5] = calib[7];  // p2
    focal_and_extra[6] = calib[8];  // k3
    focal_and_extra[7] = calib[9];  // k4
    focal_and_extra[8] = calib[10];  // sx1
    focal_and_extra[9] = calib[11];  // sy1
  }

  void SetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.SetThinPrismFisheyePoseNodesFromStackedHost(data, 0, n);
  }

  void GetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.GetThinPrismFisheyePoseNodesToStackedHost(data, 0, n);
  }

  void SetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.SetThinPrismFisheyeFocalAndExtraNodesFromStackedHost(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.GetThinPrismFisheyeFocalAndExtraNodesToStackedHost(data, 0, n);
  }

  void SetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.SetThinPrismFisheyePrincipalPointNodesFromStackedHost(data, 0, n);
  }

  void GetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.GetThinPrismFisheyePrincipalPointNodesToStackedHost(data, 0, n);
  }

  void SetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.SetThinPrismFisheyeCalibNodesFromStackedHost(data, 0, n);
  }

  void GetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.GetThinPrismFisheyeCalibNodesToStackedHost(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      // Merged variants: the calib index is the same as
      // focal_and_extra_index, so no VariantData changes are needed for
      // these cases.
      case FactorVariant::BASE:
        s.SetThinPrismFisheyeNum(n);
        s.SetThinPrismFisheyePoseIndicesFromHost(d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeCalibIndicesFromHost(d.focal_and_extra_indices.data(),
                                              n);
        s.SetThinPrismFisheyePointIndicesFromHost(d.point_indices.data(), n);
        s.SetThinPrismFisheyePixelDataFromStackedHost(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.SetThinPrismFisheyeFixedPoseNum(n);
        s.SetThinPrismFisheyeFixedPoseSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeFixedPoseCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeFixedPosePointIndicesFromHost(d.point_indices.data(),
                                                       n);
        s.SetThinPrismFisheyeFixedPosePoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeFixedPosePixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraNum(n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.SetThinPrismFisheyeSplitFixedPrincipalPointNum(n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.SetThinPrismFisheyeFixedPointNum(n);
        s.SetThinPrismFisheyeFixedPointPoseIndicesFromHost(d.pose_indices.data(),
                                                       n);
        s.SetThinPrismFisheyeFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraNum(n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointNum(n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.SetThinPrismFisheyeFixedPoseFixedPointNum(n);
        s.SetThinPrismFisheyeFixedPoseFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeFixedPoseFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeFixedPoseFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeFixedPoseFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeFixedPoseFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointNum(n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointNum(n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
            n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointNum(n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointNum(n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
            n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
    }
  }
};


// Pinhole implementation

class PinholeAdapter : public ICasparModelAdapter {
 public:
  CameraModelId ModelId() const override { return CameraModelId::kPinhole; }
  // Pinhole: params = [fx, fy, cx, cy]
  // focal           = [fx, fy]
  // principal_point = [cx, cy]
  // merged calib    = [fx, fy, cx, cy]
  size_t FocalAndExtraSize() const override { return 2; }
  size_t PrincipalPointSize() const override { return 2; }
  size_t CalibSize() const override {
    return FocalAndExtraSize() + PrincipalPointSize();
  }

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_pinhole_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        // Merged variants: both focal and principal_point are tunable.
        case FactorVariant::BASE:
          sz.num_pinhole = n;
          break;
        case FactorVariant::FIXED_POSE:
          sz.num_pinhole_fixed_pose = n;
          break;
        case FactorVariant::FIXED_POINT:
          sz.num_pinhole_fixed_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_point = n;
          break;
        // Split variants: at least one intrinsic group is fixed.
        case FactorVariant::FIXED_FOCAL_AND_EXTRA:
          sz.num_pinhole_split_fixed_focal = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_split_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
          sz.num_pinhole_split_fixed_pose_fixed_focal = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_split_fixed_pose_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_split_fixed_focal_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_pinhole_split_fixed_focal_fixed_point = n;
          break;
        case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_split_fixed_principal_point_fixed_point = n;
          break;
        case FactorVariant::
            FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_pinhole_split_fixed_pose_fixed_focal_fixed_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_split_fixed_pose_fixed_principal_point_fixed_point = n;
          break;
        case FactorVariant::
            FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_split_fixed_focal_fixed_principal_point_fixed_point =
              n;
          break;
      }
    }
  }

  void ExtractFocalAndExtra(const Camera& camera,
                            std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[0]));  // fx
    out.push_back(static_cast<StorageType>(camera.params[1]));  // fy
  }

  void ExtractPrincipalPoint(const Camera& camera,
                             std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[2]));  // cx
    out.push_back(static_cast<StorageType>(camera.params[3]));  // cy
  }

  void WriteFocalAndExtra(Camera& camera,
                          const StorageType* data,
                          size_t idx) const override {
    camera.params[0] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 0]);  // fx
    camera.params[1] =
        static_cast<double>(data[idx * FocalAndExtraSize() + 1]);  // fy
  }

  void WritePrincipalPoint(Camera& camera,
                           const StorageType* data,
                           size_t idx) const override {
    camera.params[2] =
        static_cast<double>(data[idx * PrincipalPointSize() + 0]);  // cx
    camera.params[3] =
        static_cast<double>(data[idx * PrincipalPointSize() + 1]);  // cy
  }

  void SetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.SetPinholePoseNodesFromStackedHost(data, 0, n);
  }

  void GetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.GetPinholePoseNodesToStackedHost(data, 0, n);
  }

  void SetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.SetPinholeFocalNodesFromStackedHost(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.GetPinholeFocalNodesToStackedHost(data, 0, n);
  }

  void SetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.SetPinholePrincipalPointNodesFromStackedHost(data, 0, n);
  }

  void GetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.GetPinholePrincipalPointNodesToStackedHost(data, 0, n);
  }

  void SetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.SetPinholeCalibNodesFromStackedHost(data, 0, n);
  }

  void GetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.GetPinholeCalibNodesToStackedHost(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      // Merged variants: the calib index is the same as focal_index,
      // so no VariantData changes are needed for these cases.
      case FactorVariant::BASE:
        s.SetPinholeNum(n);
        s.SetPinholePoseIndicesFromHost(d.pose_indices.data(), n);
        s.SetPinholeSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeCalibIndicesFromHost(d.focal_and_extra_indices.data(), n);
        s.SetPinholePointIndicesFromHost(d.point_indices.data(), n);
        s.SetPinholePixelDataFromStackedHost(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.SetPinholeFixedPoseNum(n);
        s.SetPinholeFixedPoseSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeFixedPoseCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPosePointIndicesFromHost(d.point_indices.data(), n);
        s.SetPinholeFixedPosePoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPosePixelDataFromStackedHost(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.SetPinholeSplitFixedFocalNum(n);
        s.SetPinholeSplitFixedFocalPoseIndicesFromHost(d.pose_indices.data(),
                                                       n);
        s.SetPinholeSplitFixedFocalSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedFocalPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeSplitFixedFocalPointIndicesFromHost(d.point_indices.data(),
                                                        n);
        s.SetPinholeSplitFixedFocalFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedFocalPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.SetPinholeSplitFixedPrincipalPointNum(n);
        s.SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.SetPinholeFixedPointNum(n);
        s.SetPinholeFixedPointPoseIndicesFromHost(d.pose_indices.data(), n);
        s.SetPinholeFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedPointPixelDataFromStackedHost(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.SetPinholeSplitFixedPoseFixedFocalNum(n);
        s.SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedFocalPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.SetPinholeSplitFixedPoseFixedPrincipalPointNum(n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.SetPinholeFixedPoseFixedPointNum(n);
        s.SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetPinholeSplitFixedFocalFixedPrincipalPointNum(n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetPinholeSplitFixedFocalFixedPointNum(n);
        s.SetPinholeSplitFixedFocalFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeSplitFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum(n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointNum(n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::
          FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
            d.sensor_from_rig_data.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
    }
  }
};

inline std::unique_ptr<ICasparModelAdapter> CreateCasparAdapter(
    const CameraModelId model_id) {
  switch (model_id) {
    case CameraModelId::kSimpleRadial:
      return std::make_unique<SimpleRadialAdapter>();
    case CameraModelId::kPinhole:
      return std::make_unique<PinholeAdapter>();
    case CameraModelId::kThinPrismFisheye:
      return std::make_unique<ThinPrismFisheyeAdapter>();
    default:
      return nullptr;
  }
}

// WARNING: Argument order is opaque and bug-prone and will change in a future
// Caspar release. Order:
//   1. Node type counts, alphabetical by type name
//   2. Factor counts, in registration order from caspar_generate.py:
//        simple_radial (4) → pinhole (4) → thin_prism_fisheye (4) →
//        simple_radial_split (11) → pinhole_split (11) →
//        thin_prism_fisheye_split (11)
inline caspar::GraphSolver CreateSolver(
    const caspar::SolverParams<StorageType>& params,
    const CasparSolverSizing& sz,
    size_t device_id = 0) {
  return caspar::GraphSolver(
      params,
      // Node type counts (alphabetical):
      //   PinholeCalib, PinholeFocal, PinholePose,
      //   PinholePrincipalPoint, Point,
      //   SimpleRadialCalib, SimpleRadialFocalAndExtra,
      //   SimpleRadialPose, SimpleRadialPrincipalPoint,
      //   ThinPrismFisheyeCalib, ThinPrismFisheyeFocalAndExtra,
      //   ThinPrismFisheyePose, ThinPrismFisheyePrincipalPoint
      sz.num_pinhole_calibs,        // PinholeCalib        (merged pool)
      sz.num_pinhole_calibs,        // PinholeFocal         (split pool)
      sz.num_pinhole_poses,         // PinholePose
      sz.num_pinhole_calibs,        // PinholePrincipalPoint (split pool)
      sz.num_points,                // Point
      sz.num_simple_radial_calibs,  // SimpleRadialCalib              (merged
                                    // pool)
      sz.num_simple_radial_calibs,  // SimpleRadialFocalAndExtra  (split
                                    // pool)
      sz.num_simple_radial_poses,   // SimpleRadialPose
      sz.num_simple_radial_calibs,  // SimpleRadialPrincipalPoint      (split
                                    // pool)
      sz.num_thin_prism_fisheye_calibs,  // ThinPrismFisheyeCalib       (merged
                                         // pool)
      sz.num_thin_prism_fisheye_calibs,  // ThinPrismFisheyeFocalAndExtra
                                         // (split pool)
      sz.num_thin_prism_fisheye_poses,   // ThinPrismFisheyePose
      sz.num_thin_prism_fisheye_calibs,  // ThinPrismFisheyePrincipalPoint
                                         // (split pool)
      // simple_radial factor counts (r=0..2 over {pose, point}):
      sz.num_simple_radial,                         // {}
      sz.num_simple_radial_fixed_pose,              // {pose}
      sz.num_simple_radial_fixed_point,             // {point}
      sz.num_simple_radial_fixed_pose_fixed_point,  // {pose, point}
      // pinhole factor counts (same order):
      sz.num_pinhole,                         // {}
      sz.num_pinhole_fixed_pose,              // {pose}
      sz.num_pinhole_fixed_point,             // {point}
      sz.num_pinhole_fixed_pose_fixed_point,  // {pose, point}
      // thin_prism_fisheye factor counts (same order):
      sz.num_thin_prism_fisheye,                         // {}
      sz.num_thin_prism_fisheye_fixed_pose,              // {pose}
      sz.num_thin_prism_fisheye_fixed_point,             // {point}
      sz.num_thin_prism_fisheye_fixed_pose_fixed_point,  // {pose, point}
      // simple_radial_split factor counts (11 variants, must_fix_one_of):
      sz.num_simple_radial_split_fixed_focal_and_extra,             // r=1 {fae}
      sz.num_simple_radial_split_fixed_principal_point,             // r=1 {pp}
      sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra,  // r=2
                                                                    // {pose,fad}
      sz.num_simple_radial_split_fixed_pose_fixed_principal_point,  // r=2
                                                                    // {pose,pp}
      sz.num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point,  // r=2 {fae,pp}
      sz.num_simple_radial_split_fixed_focal_and_extra_fixed_point,  // r=2
                                                                     // {fae,pt}
      sz.num_simple_radial_split_fixed_principal_point_fixed_point,  // r=2
                                                                     // {pp,pt}
      sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point,  // r=3
      sz.num_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point,  // r=3
      sz.num_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point,  // r=3
      sz.num_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point,  // r=3
      // pinhole_split factor counts (same 11-variant order):
      sz.num_pinhole_split_fixed_focal,                        // r=1 {f}
      sz.num_pinhole_split_fixed_principal_point,              // r=1 {pp}
      sz.num_pinhole_split_fixed_pose_fixed_focal,             // r=2 {pose,f}
      sz.num_pinhole_split_fixed_pose_fixed_principal_point,   // r=2 {pose,pp}
      sz.num_pinhole_split_fixed_focal_fixed_principal_point,  // r=2 {f,pp}
      sz.num_pinhole_split_fixed_focal_fixed_point,            // r=2 {f,pt}
      sz.num_pinhole_split_fixed_principal_point_fixed_point,  // r=2 {pp,pt}
      sz.num_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point,  // r=3
      sz.num_pinhole_split_fixed_pose_fixed_focal_fixed_point,            // r=3
      sz.num_pinhole_split_fixed_pose_fixed_principal_point_fixed_point,  // r=3
      sz.num_pinhole_split_fixed_focal_fixed_principal_point_fixed_point,  // r=3
      // thin_prism_fisheye_split factor counts (same 11-variant order):
      sz.num_thin_prism_fisheye_split_fixed_focal_and_extra,  // r=1 {fae}
      sz.num_thin_prism_fisheye_split_fixed_principal_point,  // r=1 {pp}
      sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra,  // r=2
                                                                         // {pose,fae}
      sz.num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point,  // r=2
                                                                         // {pose,pp}
      sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point,  // r=2 {fae,pp}
      sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point,  // r=2
                                                                          // {fae,pt}
      sz.num_thin_prism_fisheye_split_fixed_principal_point_fixed_point,  // r=2
                                                                          // {pp,pt}
      sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point,  // r=3
      sz.num_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point,  // r=3
      sz.num_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point,  // r=3
      sz.num_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point,  // r=3
      device_id);
}

}  // namespace colmap
