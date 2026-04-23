// caspar_model_adapter.h
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

  // SimpleRadial — num_calibs is shared by the merged Calib pool and the split
  // FocalAndExtra / PrincipalPoint pools (all sized 1:1 with cameras).
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

  // Pinhole — same layout as SimpleRadial above.
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

// Interface — one implementation per camera model.

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

  // Extract focal_and_extra / principal_point params from a camera into a flat
  // output vector.
  virtual void ExtractFocalAndExtra(const Camera& camera,
                                    std::vector<StorageType>& out) const = 0;
  virtual void ExtractPrincipalPoint(const Camera& camera,
                                     std::vector<StorageType>& out) const = 0;

  // Write optimized focal_and_extra / principal_point back into a camera's
  // params.
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
    s.set_SimpleRadialPose_nodes_from_stacked_host(data, 0, n);
  }

  void GetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.get_SimpleRadialPose_nodes_to_stacked_host(data, 0, n);
  }

  void SetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.set_SimpleRadialFocalAndExtra_nodes_from_stacked_host(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.get_SimpleRadialFocalAndExtra_nodes_to_stacked_host(data, 0, n);
  }

  void SetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.set_SimpleRadialPrincipalPoint_nodes_from_stacked_host(data, 0, n);
  }

  void GetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.get_SimpleRadialPrincipalPoint_nodes_to_stacked_host(data, 0, n);
  }

  void SetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.set_SimpleRadialCalib_nodes_from_stacked_host(data, 0, n);
  }

  void GetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.get_SimpleRadialCalib_nodes_to_stacked_host(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      // Merged variants: calib indices reuse focal_and_extra_indices
      // (same camera index value — no VariantData changes needed).
      case FactorVariant::BASE:
        s.set_simple_radial_merged_num(n);
        s.set_simple_radial_merged_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_merged_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_merged_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_merged_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.set_simple_radial_merged_fixed_pose_num(n);
        s.set_simple_radial_merged_fixed_pose_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_merged_fixed_pose_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_merged_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.set_simple_radial_fixed_focal_and_extra_num(n);
        s.set_simple_radial_fixed_focal_and_extra_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.set_simple_radial_fixed_principal_point_num(n);
        s.set_simple_radial_fixed_principal_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.set_simple_radial_merged_fixed_point_num(n);
        s.set_simple_radial_merged_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_merged_fixed_point_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_merged_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_merged_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_num(n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.set_simple_radial_fixed_pose_fixed_principal_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_simple_radial_merged_fixed_pose_fixed_point_num(n);
        s.set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_merged_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_merged_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_merged_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num(n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_num(n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_simple_radial_fixed_principal_point_fixed_point_num(n);
        s.set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
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
    s.set_PinholePose_nodes_from_stacked_host(data, 0, n);
  }

  void GetPoseNodes(caspar::GraphSolver& s,
                    StorageType* data,
                    size_t n) const override {
    s.get_PinholePose_nodes_to_stacked_host(data, 0, n);
  }

  void SetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.set_PinholeFocalAndExtra_nodes_from_stacked_host(data, 0, n);
  }

  void GetFocalAndExtraNodes(caspar::GraphSolver& s,
                             StorageType* data,
                             size_t n) const override {
    s.get_PinholeFocalAndExtra_nodes_to_stacked_host(data, 0, n);
  }

  void SetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.set_PinholePrincipalPoint_nodes_from_stacked_host(data, 0, n);
  }

  void GetPrincipalPointNodes(caspar::GraphSolver& s,
                              StorageType* data,
                              size_t n) const override {
    s.get_PinholePrincipalPoint_nodes_to_stacked_host(data, 0, n);
  }

  void SetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.set_PinholeCalib_nodes_from_stacked_host(data, 0, n);
  }

  void GetCalibNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.get_PinholeCalib_nodes_to_stacked_host(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      // Merged variants: calib indices reuse focal_and_extra_indices
      // (same camera index value — no VariantData changes needed).
      case FactorVariant::BASE:
        s.set_pinhole_merged_num(n);
        s.set_pinhole_merged_pose_indices_from_host(d.pose_indices.data(), n);
        s.set_pinhole_merged_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_merged_point_indices_from_host(d.point_indices.data(), n);
        s.set_pinhole_merged_pixel_data_from_stacked_host(d.pixels.data(), 0,
                                                          n);
        break;
      case FactorVariant::FIXED_POSE:
        s.set_pinhole_merged_fixed_pose_num(n);
        s.set_pinhole_merged_fixed_pose_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_merged_fixed_pose_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_merged_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_merged_fixed_pose_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        s.set_pinhole_fixed_focal_and_extra_num(n);
        s.set_pinhole_fixed_focal_and_extra_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        s.set_pinhole_fixed_principal_point_num(n);
        s.set_pinhole_fixed_principal_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_principal_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.set_pinhole_merged_fixed_point_num(n);
        s.set_pinhole_merged_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_merged_fixed_point_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_merged_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_merged_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_num(n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        s.set_pinhole_fixed_pose_fixed_principal_point_num(n);
        s.set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_pinhole_merged_fixed_pose_fixed_point_num(n);
        s.set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_merged_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_num(n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.set_pinhole_fixed_focal_and_extra_fixed_point_num(n);
        s.set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_pinhole_fixed_principal_point_fixed_point_num(n);
        s.set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num(n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
            d.principal_point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num(n);
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
            d.focal_and_extra_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
            d.const_focal_and_extra.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
            d.const_principal_point.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
    }
  }
};

// Factory
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

// Solver factory
//
// WARNING: This call is very tedious and bug-prone, and will change in a
// newer release of Caspar. Argument order:
//   1. Node type counts — alphabetical by type name
//   2. Factor counts — in registration order from caspar_generate.py:
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
