#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"

#include <memory>
#include <vector>

#include <solver.h>

namespace colmap {

struct CasparSolverSizing {
  // Pose pools are per-model to prevent cross-model factor batching.
  size_t num_simple_radial_poses = 0;
  size_t num_pinhole_poses = 0;
  size_t num_points = 0;

  // SimpleRadial: num_calibs is shared by the merged Calib pool and the split
  // FocalAndExtra / PrincipalPoint pools, one entry per camera.
  // Merged variants (both intrinsic groups tunable): 4 counts below.
  // Split variants (at least one group fixed): 11 counts below.
  size_t num_simple_radial_calibs = 0;
  size_t num_simple_radial_merged = 0;
  size_t num_simple_radial_merged_fixed_pose = 0;
  size_t num_simple_radial_merged_fixed_point = 0;
  size_t num_simple_radial_merged_fixed_pose_fixed_point = 0;
  size_t num_simple_radial_fixed_focal_and_extra = 0;
  size_t num_simple_radial_fixed_principal_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_focal_and_extra = 0;
  size_t num_simple_radial_fixed_pose_fixed_principal_point = 0;
  size_t num_simple_radial_fixed_focal_and_extra_fixed_principal_point = 0;
  size_t num_simple_radial_fixed_focal_and_extra_fixed_point = 0;
  size_t num_simple_radial_fixed_principal_point_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_principal_point_fixed_point = 0;
  size_t num_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point = 0;

  // Pinhole: same layout as SimpleRadial above.
  size_t num_pinhole_calibs = 0;
  size_t num_pinhole_merged = 0;
  size_t num_pinhole_merged_fixed_pose = 0;
  size_t num_pinhole_merged_fixed_point = 0;
  size_t num_pinhole_merged_fixed_pose_fixed_point = 0;
  size_t num_pinhole_fixed_focal_and_extra = 0;
  size_t num_pinhole_fixed_principal_point = 0;
  size_t num_pinhole_fixed_pose_fixed_focal_and_extra = 0;
  size_t num_pinhole_fixed_pose_fixed_principal_point = 0;
  size_t num_pinhole_fixed_focal_and_extra_fixed_principal_point = 0;
  size_t num_pinhole_fixed_focal_and_extra_fixed_point = 0;
  size_t num_pinhole_fixed_principal_point_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point = 0;
  size_t num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_principal_point_fixed_point = 0;
  size_t num_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point = 0;
};

// One implementation per camera model.

class ICasparModelAdapter {
 public:
  virtual ~ICasparModelAdapter() = default;

  virtual CameraModelId ModelId() const = 0;

  // Number of floats in the focal_and_extra and principal_point node arrays
  // per camera.
  virtual size_t FocalAndExtraSize() const = 0;
  virtual size_t PrincipalPointSize() const = 0;
  // Number of floats in the merged Calib node per camera
  // (= FocalAndExtraSize() + PrincipalPointSize()).
  virtual size_t CalibSize() const = 0;

  virtual void SetCalibNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;
  virtual void GetCalibNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;

  virtual void FillSizing(CasparSolverSizing& sz,
                          const ModelData& md,
                          size_t num_calibs) const = 0;

  // Append focal_and_extra / principal_point params from a camera into a flat
  // output vector.
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
  // principal_point = [cx, cy]
  // merged calib    = [f, k, cx, cy]
  size_t FocalAndExtraSize() const override { return 2; }
  size_t PrincipalPointSize() const override { return 2; }
  size_t CalibSize() const override { return FocalAndExtraSize() + PrincipalPointSize(); }

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_simple_radial_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        // Merged variants: both focal_and_extra and principal_point are tunable.
        case FactorVariant::BASE:
          sz.num_simple_radial_merged = n; break;
        case FactorVariant::FIXED_POSE:
          sz.num_simple_radial_merged_fixed_pose = n; break;
        case FactorVariant::FIXED_POINT:
          sz.num_simple_radial_merged_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_simple_radial_merged_fixed_pose_fixed_point = n; break;
        // Split variants: at least one intrinsic group is fixed.
        case FactorVariant::FIXED_FOCAL_AND_EXTRA:
          sz.num_simple_radial_fixed_focal_and_extra = n; break;
        case FactorVariant::FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_fixed_principal_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
          sz.num_simple_radial_fixed_pose_fixed_focal_and_extra = n; break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_fixed_pose_fixed_principal_point = n; break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_fixed_focal_and_extra_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_simple_radial_fixed_focal_and_extra_fixed_point = n; break;
        case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_fixed_principal_point_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_principal_point_fixed_point = n;
          break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point = n;
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
      // Merged variants: the calib index is the same as focal_and_extra_index,
      // so no VariantData changes are needed for these cases.
      case FactorVariant::BASE:
        s.SetSimpleRadialMergedNum(n);
        s.SetSimpleRadialMergedPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialMergedCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialMergedPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialMergedPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.SetSimpleRadialMergedFixedPoseNum(n);
        s.SetSimpleRadialMergedFixedPoseCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialMergedFixedPosePointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialMergedFixedPosePoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialMergedFixedPosePixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.SetSimpleRadialFixedFocalAndExtraNum(n);
        s.SetSimpleRadialFixedFocalAndExtraPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialFixedPrincipalPointNum(n);
        s.SetSimpleRadialFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.SetSimpleRadialMergedFixedPointNum(n);
        s.SetSimpleRadialMergedFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialMergedFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialMergedFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialMergedFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraNum(n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialFixedPoseFixedPrincipalPointNum(n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.SetSimpleRadialMergedFixedPoseFixedPointNum(n);
        s.SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetSimpleRadialFixedFocalAndExtraFixedPointNum(n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialFixedPrincipalPointFixedPointNum(n);
        s.SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointNum(n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointNum(n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointNum(n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
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
  // focal_and_extra = [fx, fy]
  // principal_point = [cx, cy]
  // merged calib    = [fx, fy, cx, cy]
  size_t FocalAndExtraSize() const override { return 2; }
  size_t PrincipalPointSize() const override { return 2; }
  size_t CalibSize() const override { return FocalAndExtraSize() + PrincipalPointSize(); }

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_pinhole_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        // Merged variants: both focal_and_extra and principal_point are tunable.
        case FactorVariant::BASE:
          sz.num_pinhole_merged = n; break;
        case FactorVariant::FIXED_POSE:
          sz.num_pinhole_merged_fixed_pose = n; break;
        case FactorVariant::FIXED_POINT:
          sz.num_pinhole_merged_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_pinhole_merged_fixed_pose_fixed_point = n; break;
        // Split variants: at least one intrinsic group is fixed.
        case FactorVariant::FIXED_FOCAL_AND_EXTRA:
          sz.num_pinhole_fixed_focal_and_extra = n; break;
        case FactorVariant::FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_fixed_principal_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
          sz.num_pinhole_fixed_pose_fixed_focal_and_extra = n; break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_fixed_pose_fixed_principal_point = n; break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_fixed_focal_and_extra_fixed_principal_point = n; break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_pinhole_fixed_focal_and_extra_fixed_point = n; break;
        case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_fixed_principal_point_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
          sz.num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point = n;
          break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_principal_point_fixed_point = n; break;
        case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
          sz.num_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point = n;
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
    s.SetPinholeFocalAndExtraNodesFromStackedHost(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.GetPinholeFocalAndExtraNodesToStackedHost(data, 0, n);
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
      // Merged variants: the calib index is the same as focal_and_extra_index,
      // so no VariantData changes are needed for these cases.
      case FactorVariant::BASE:
        s.SetPinholeMergedNum(n);
        s.SetPinholeMergedPoseIndicesFromHost(d.pose_indices.data(), n);
        s.SetPinholeMergedCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeMergedPointIndicesFromHost(d.point_indices.data(), n);
        s.SetPinholeMergedPixelDataFromStackedHost(d.pixels.data(), 0,
                                                          n);
        break;
      case FactorVariant::FIXED_POSE:
        s.SetPinholeMergedFixedPoseNum(n);
        s.SetPinholeMergedFixedPoseCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeMergedFixedPosePointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeMergedFixedPosePoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeMergedFixedPosePixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.SetPinholeFixedFocalAndExtraNum(n);
        s.SetPinholeFixedFocalAndExtraPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.SetPinholeFixedPrincipalPointNum(n);
        s.SetPinholeFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.SetPinholeMergedFixedPointNum(n);
        s.SetPinholeMergedFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeMergedFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeMergedFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeMergedFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.SetPinholeFixedPoseFixedFocalAndExtraNum(n);
        s.SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.SetPinholeFixedPoseFixedPrincipalPointNum(n);
        s.SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.SetPinholeMergedFixedPoseFixedPointNum(n);
        s.SetPinholeMergedFixedPoseFixedPointCalibIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeMergedFixedPoseFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetPinholeFixedFocalAndExtraFixedPointNum(n);
        s.SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
            d.point_indices.data(), n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointNum(n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
            d.principal_point_indices.data(), n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
            d.focal_and_extra_indices.data(), n);
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
            d.const_poses.data(), 0, n);
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointNum(n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
            d.pose_indices.data(), n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
            d.const_focal_and_extra.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
            d.const_principal_point.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
            d.const_points.data(), 0, n);
        s.SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
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
    default:
      return nullptr;
  }
}

// WARNING: Argument order is opaque and bug-prone and will change in a future
// Caspar release. Order:
//   1. Node type counts, alphabetical by type name
//   2. Factor counts, in registration order from caspar_generate.py:
//        simple_radial_merged (4) → pinhole_merged (4) →
//        simple_radial split (11) → pinhole split (11)
inline caspar::GraphSolver CreateSolver(
    const caspar::SolverParams<StorageType>& params,
    const CasparSolverSizing& sz) {
  return caspar::GraphSolver(
      params,
      // Node type counts (alphabetical):
      //   PinholeCalib, PinholeFocalAndExtra, PinholePose,
      //   PinholePrincipalPoint, Point,
      //   SimpleRadialCalib, SimpleRadialFocalAndExtra,
      //   SimpleRadialPose, SimpleRadialPrincipalPoint
      sz.num_pinhole_calibs,          // PinholeCalib        (merged pool)
      sz.num_pinhole_calibs,          // PinholeFocalAndExtra (split pool)
      sz.num_pinhole_poses,           // PinholePose
      sz.num_pinhole_calibs,          // PinholePrincipalPoint (split pool)
      sz.num_points,                  // Point
      sz.num_simple_radial_calibs,    // SimpleRadialCalib        (merged pool)
      sz.num_simple_radial_calibs,    // SimpleRadialFocalAndExtra (split pool)
      sz.num_simple_radial_poses,     // SimpleRadialPose
      sz.num_simple_radial_calibs,    // SimpleRadialPrincipalPoint (split pool)
      // simple_radial_merged factor counts (r=0..2 over {pose, point}):
      sz.num_simple_radial_merged,                          // {}
      sz.num_simple_radial_merged_fixed_pose,               // {pose}
      sz.num_simple_radial_merged_fixed_point,              // {point}
      sz.num_simple_radial_merged_fixed_pose_fixed_point,   // {pose, point}
      // pinhole_merged factor counts (same order):
      sz.num_pinhole_merged,                                // {}
      sz.num_pinhole_merged_fixed_pose,                     // {pose}
      sz.num_pinhole_merged_fixed_point,                    // {point}
      sz.num_pinhole_merged_fixed_pose_fixed_point,         // {pose, point}
      // simple_radial split factor counts (11 variants, must_fix_one_of):
      sz.num_simple_radial_fixed_focal_and_extra,                               // r=1 {fae}
      sz.num_simple_radial_fixed_principal_point,                               // r=1 {pp}
      sz.num_simple_radial_fixed_pose_fixed_focal_and_extra,                    // r=2 {pose,fae}
      sz.num_simple_radial_fixed_pose_fixed_principal_point,                    // r=2 {pose,pp}
      sz.num_simple_radial_fixed_focal_and_extra_fixed_principal_point,         // r=2 {fae,pp}
      sz.num_simple_radial_fixed_focal_and_extra_fixed_point,                   // r=2 {fae,pt}
      sz.num_simple_radial_fixed_principal_point_fixed_point,                   // r=2 {pp,pt}
      sz.num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point,  // r=3
      sz.num_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point,            // r=3
      sz.num_simple_radial_fixed_pose_fixed_principal_point_fixed_point,            // r=3
      sz.num_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point, // r=3
      // pinhole split factor counts (same 11-variant order):
      sz.num_pinhole_fixed_focal_and_extra,                               // r=1 {fae}
      sz.num_pinhole_fixed_principal_point,                               // r=1 {pp}
      sz.num_pinhole_fixed_pose_fixed_focal_and_extra,                    // r=2 {pose,fae}
      sz.num_pinhole_fixed_pose_fixed_principal_point,                    // r=2 {pose,pp}
      sz.num_pinhole_fixed_focal_and_extra_fixed_principal_point,         // r=2 {fae,pp}
      sz.num_pinhole_fixed_focal_and_extra_fixed_point,                   // r=2 {fae,pt}
      sz.num_pinhole_fixed_principal_point_fixed_point,                   // r=2 {pp,pt}
      sz.num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point,  // r=3
      sz.num_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point,            // r=3
      sz.num_pinhole_fixed_pose_fixed_principal_point_fixed_point,            // r=3
      sz.num_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point  // r=3
  );
}

}  // namespace colmap
