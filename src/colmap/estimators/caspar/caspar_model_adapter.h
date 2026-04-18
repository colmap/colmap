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
  size_t num_poses = 0;
  size_t num_points = 0;

  // SimpleRadial — num_calibs counts both focal and extra_calib nodes (1:1).
  // 12 of the 15 generated variants are dispatched; the 3 with both focal and
  // extra_calib fixed are reserved (0-sized).
  size_t num_simple_radial_calibs = 0;
  size_t num_simple_radial = 0;
  size_t num_simple_radial_fixed_pose = 0;
  size_t num_simple_radial_fixed_focal = 0;
  size_t num_simple_radial_fixed_extra_calib = 0;
  size_t num_simple_radial_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_focal = 0;
  size_t num_simple_radial_fixed_pose_fixed_extra_calib = 0;
  size_t num_simple_radial_fixed_pose_fixed_point = 0;
  size_t num_simple_radial_fixed_focal_fixed_extra_calib = 0;   // reserved
  size_t num_simple_radial_fixed_focal_fixed_point = 0;
  size_t num_simple_radial_fixed_extra_calib_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib = 0;  // reserved
  size_t num_simple_radial_fixed_pose_fixed_focal_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_extra_calib_fixed_point = 0;
  size_t num_simple_radial_fixed_focal_fixed_extra_calib_fixed_point = 0;  // reserved

  // Pinhole — same layout as SimpleRadial above.
  size_t num_pinhole_calibs = 0;
  size_t num_pinhole = 0;
  size_t num_pinhole_fixed_pose = 0;
  size_t num_pinhole_fixed_focal = 0;
  size_t num_pinhole_fixed_extra_calib = 0;
  size_t num_pinhole_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_focal = 0;
  size_t num_pinhole_fixed_pose_fixed_extra_calib = 0;
  size_t num_pinhole_fixed_pose_fixed_point = 0;
  size_t num_pinhole_fixed_focal_fixed_extra_calib = 0;   // reserved
  size_t num_pinhole_fixed_focal_fixed_point = 0;
  size_t num_pinhole_fixed_extra_calib_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_focal_fixed_extra_calib = 0;  // reserved
  size_t num_pinhole_fixed_pose_fixed_focal_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_extra_calib_fixed_point = 0;
  size_t num_pinhole_fixed_focal_fixed_extra_calib_fixed_point = 0;  // reserved
};

// Interface — one implementation per camera model.

class ICasparModelAdapter {
 public:
  virtual ~ICasparModelAdapter() = default;

  virtual CameraModelId ModelId() const = 0;

  // Number of floats in the focal and extra_calib node arrays per camera.
  virtual size_t FocalSize() const = 0;
  virtual size_t ExtraCalibSize() const = 0;

  virtual void FillSizing(CasparSolverSizing& sz,
                          const ModelData& md,
                          size_t num_calibs) const = 0;

  // Extract focal / extra_calib params from a camera into a flat output vector.
  virtual void ExtractFocal(const Camera& camera,
                             std::vector<StorageType>& out) const = 0;
  virtual void ExtractExtraCalib(const Camera& camera,
                                 std::vector<StorageType>& out) const = 0;

  // Write optimized focal / extra_calib back into a camera's params.
  virtual void WriteFocal(Camera& camera,
                          const StorageType* focal_data,
                          size_t idx) const = 0;
  virtual void WriteExtraCalib(Camera& camera,
                               const StorageType* extra_calib_data,
                               size_t idx) const = 0;

  virtual void SetFocalNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;
  virtual void GetFocalNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;
  virtual void SetExtraCalibNodes(caspar::GraphSolver& solver,
                                  StorageType* data,
                                  size_t n) const = 0;
  virtual void GetExtraCalibNodes(caspar::GraphSolver& solver,
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
  size_t FocalSize() const override { return 1; }      // [f]
  size_t ExtraCalibSize() const override { return 3; } // [cx, cy, k]

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_simple_radial_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        case FactorVariant::BASE:
          sz.num_simple_radial = n; break;
        case FactorVariant::FIXED_POSE:
          sz.num_simple_radial_fixed_pose = n; break;
        case FactorVariant::FIXED_FOCAL:
          sz.num_simple_radial_fixed_focal = n; break;
        case FactorVariant::FIXED_EXTRA_CALIB:
          sz.num_simple_radial_fixed_extra_calib = n; break;
        case FactorVariant::FIXED_POINT:
          sz.num_simple_radial_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL:
          sz.num_simple_radial_fixed_pose_fixed_focal = n; break;
        case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB:
          sz.num_simple_radial_fixed_pose_fixed_extra_calib = n; break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_point = n; break;
        case FactorVariant::FIXED_FOCAL_FIXED_POINT:
          sz.num_simple_radial_fixed_focal_fixed_point = n; break;
        case FactorVariant::FIXED_EXTRA_CALIB_FIXED_POINT:
          sz.num_simple_radial_fixed_extra_calib_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_focal_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_extra_calib_fixed_point = n;
          break;
      }
    }
  }

  void ExtractFocal(const Camera& camera,
                    std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[0]));  // f
  }

  void ExtractExtraCalib(const Camera& camera,
                         std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[1]));  // cx
    out.push_back(static_cast<StorageType>(camera.params[2]));  // cy
    out.push_back(static_cast<StorageType>(camera.params[3]));  // k
  }

  void WriteFocal(Camera& camera,
                  const StorageType* focal_data,
                  size_t idx) const override {
    camera.params[0] = static_cast<double>(focal_data[idx * FocalSize()]);
  }

  void WriteExtraCalib(Camera& camera,
                       const StorageType* extra_calib_data,
                       size_t idx) const override {
    camera.params[1] =
        static_cast<double>(extra_calib_data[idx * ExtraCalibSize() + 0]);
    camera.params[2] =
        static_cast<double>(extra_calib_data[idx * ExtraCalibSize() + 1]);
    camera.params[3] =
        static_cast<double>(extra_calib_data[idx * ExtraCalibSize() + 2]);
  }

  void SetFocalNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.set_SimpleRadialFocal_nodes_from_stacked_host(data, 0, n);
  }

  void GetFocalNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.get_SimpleRadialFocal_nodes_to_stacked_host(data, 0, n);
  }

  void SetExtraCalibNodes(caspar::GraphSolver& s,
                          StorageType* data,
                          size_t n) const override {
    s.set_SimpleRadialExtraCalib_nodes_from_stacked_host(data, 0, n);
  }

  void GetExtraCalibNodes(caspar::GraphSolver& s,
                          StorageType* data,
                          size_t n) const override {
    s.get_SimpleRadialExtraCalib_nodes_to_stacked_host(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      case FactorVariant::BASE:
        s.set_simple_radial_num(n);
        s.set_simple_radial_pose_indices_from_host(d.pose_indices.data(), n);
        s.set_simple_radial_focal_indices_from_host(d.focal_indices.data(), n);
        s.set_simple_radial_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_point_indices_from_host(d.point_indices.data(), n);
        s.set_simple_radial_pixel_data_from_stacked_host(d.pixels.data(), 0,
                                                         n);
        break;
      case FactorVariant::FIXED_POSE:
        s.set_simple_radial_fixed_pose_num(n);
        s.set_simple_radial_fixed_pose_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_pose_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL:
        s.set_simple_radial_fixed_focal_num(n);
        s.set_simple_radial_fixed_focal_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_focal_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_focal_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_simple_radial_fixed_focal_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_EXTRA_CALIB:
        s.set_simple_radial_fixed_extra_calib_num(n);
        s.set_simple_radial_fixed_extra_calib_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_extra_calib_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_extra_calib_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.set_simple_radial_fixed_point_num(n);
        s.set_simple_radial_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL:
        s.set_simple_radial_fixed_pose_fixed_focal_num(n);
        s.set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB:
        s.set_simple_radial_fixed_pose_fixed_extra_calib_num(n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_FIXED_POINT:
        s.set_simple_radial_fixed_focal_fixed_point_num(n);
        s.set_simple_radial_fixed_focal_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_EXTRA_CALIB_FIXED_POINT:
        s.set_simple_radial_fixed_extra_calib_fixed_point_num(n);
        s.set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num(n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
    }
  }
};

// Pinhole implementation

class PinholeAdapter : public ICasparModelAdapter {
 public:
  CameraModelId ModelId() const override { return CameraModelId::kPinhole; }
  size_t FocalSize() const override { return 2; }      // [fx, fy]
  size_t ExtraCalibSize() const override { return 2; } // [cx, cy]

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_pinhole_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      const size_t n = md.variants[v].num_factors;
      switch (static_cast<FactorVariant>(v)) {
        case FactorVariant::BASE:
          sz.num_pinhole = n; break;
        case FactorVariant::FIXED_POSE:
          sz.num_pinhole_fixed_pose = n; break;
        case FactorVariant::FIXED_FOCAL:
          sz.num_pinhole_fixed_focal = n; break;
        case FactorVariant::FIXED_EXTRA_CALIB:
          sz.num_pinhole_fixed_extra_calib = n; break;
        case FactorVariant::FIXED_POINT:
          sz.num_pinhole_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL:
          sz.num_pinhole_fixed_pose_fixed_focal = n; break;
        case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB:
          sz.num_pinhole_fixed_pose_fixed_extra_calib = n; break;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_point = n; break;
        case FactorVariant::FIXED_FOCAL_FIXED_POINT:
          sz.num_pinhole_fixed_focal_fixed_point = n; break;
        case FactorVariant::FIXED_EXTRA_CALIB_FIXED_POINT:
          sz.num_pinhole_fixed_extra_calib_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_FOCAL_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_focal_fixed_point = n; break;
        case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_extra_calib_fixed_point = n; break;
      }
    }
  }

  void ExtractFocal(const Camera& camera,
                    std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[0]));  // fx
    out.push_back(static_cast<StorageType>(camera.params[1]));  // fy
  }

  void ExtractExtraCalib(const Camera& camera,
                         std::vector<StorageType>& out) const override {
    out.push_back(static_cast<StorageType>(camera.params[2]));  // cx
    out.push_back(static_cast<StorageType>(camera.params[3]));  // cy
  }

  void WriteFocal(Camera& camera,
                  const StorageType* focal_data,
                  size_t idx) const override {
    camera.params[0] =
        static_cast<double>(focal_data[idx * FocalSize() + 0]);
    camera.params[1] =
        static_cast<double>(focal_data[idx * FocalSize() + 1]);
  }

  void WriteExtraCalib(Camera& camera,
                       const StorageType* extra_calib_data,
                       size_t idx) const override {
    camera.params[2] =
        static_cast<double>(extra_calib_data[idx * ExtraCalibSize() + 0]);
    camera.params[3] =
        static_cast<double>(extra_calib_data[idx * ExtraCalibSize() + 1]);
  }

  void SetFocalNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.set_PinholeFocal_nodes_from_stacked_host(data, 0, n);
  }

  void GetFocalNodes(caspar::GraphSolver& s,
                     StorageType* data,
                     size_t n) const override {
    s.get_PinholeFocal_nodes_to_stacked_host(data, 0, n);
  }

  void SetExtraCalibNodes(caspar::GraphSolver& s,
                          StorageType* data,
                          size_t n) const override {
    s.set_PinholeExtraCalib_nodes_from_stacked_host(data, 0, n);
  }

  void GetExtraCalibNodes(caspar::GraphSolver& s,
                          StorageType* data,
                          size_t n) const override {
    s.get_PinholeExtraCalib_nodes_to_stacked_host(data, 0, n);
  }

  void SetVariantFactors(caspar::GraphSolver& s,
                         FactorVariant variant,
                         const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      case FactorVariant::BASE:
        s.set_pinhole_num(n);
        s.set_pinhole_pose_indices_from_host(d.pose_indices.data(), n);
        s.set_pinhole_focal_indices_from_host(d.focal_indices.data(), n);
        s.set_pinhole_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_point_indices_from_host(d.point_indices.data(), n);
        s.set_pinhole_pixel_data_from_stacked_host(d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE:
        s.set_pinhole_fixed_pose_num(n);
        s.set_pinhole_fixed_pose_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_pose_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_pose_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_pixel_data_from_stacked_host(d.pixels.data(),
                                                              0, n);
        break;
      case FactorVariant::FIXED_FOCAL:
        s.set_pinhole_fixed_focal_num(n);
        s.set_pinhole_fixed_focal_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_focal_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_focal_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_pinhole_fixed_focal_pixel_data_from_stacked_host(d.pixels.data(),
                                                               0, n);
        break;
      case FactorVariant::FIXED_EXTRA_CALIB:
        s.set_pinhole_fixed_extra_calib_num(n);
        s.set_pinhole_fixed_extra_calib_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_extra_calib_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_extra_calib_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_pinhole_fixed_extra_calib_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POINT:
        s.set_pinhole_fixed_point_num(n);
        s.set_pinhole_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_point_pixel_data_from_stacked_host(d.pixels.data(),
                                                               0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL:
        s.set_pinhole_fixed_pose_fixed_focal_num(n);
        s.set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB:
        s.set_pinhole_fixed_pose_fixed_extra_calib_num(n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_point_num(n);
        s.set_pinhole_fixed_pose_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_FOCAL_FIXED_POINT:
        s.set_pinhole_fixed_focal_fixed_point_num(n);
        s.set_pinhole_fixed_focal_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_EXTRA_CALIB_FIXED_POINT:
        s.set_pinhole_fixed_extra_calib_fixed_point_num(n);
        s.set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_num(n);
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
            d.extra_calib_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
            d.const_focal.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        break;
      case FactorVariant::FIXED_POSE_FIXED_EXTRA_CALIB_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num(n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
            d.focal_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
            d.const_extra_calib.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
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
//   2. Factor counts — in definition order from caspar_generate.py (simple_radial
//      variants first, then pinhole; within each model, subset order r=0..3)
inline caspar::GraphSolver CreateSolver(
    const caspar::SolverParams<StorageType>& params,
    const CasparSolverSizing& sz) {
  return caspar::GraphSolver(
      params,
      // Node type counts (alphabetical):
      //   PinholeExtraCalib, PinholeFocal, Point, Pose,
      //   SimpleRadialExtraCalib, SimpleRadialFocal
      sz.num_pinhole_calibs,          // PinholeExtraCalib
      sz.num_pinhole_calibs,          // PinholeFocal
      sz.num_points,                  // Point
      sz.num_poses,                   // Pose
      sz.num_simple_radial_calibs,    // SimpleRadialExtraCalib
      sz.num_simple_radial_calibs,    // SimpleRadialFocal
      // simple_radial factor counts (r=0..3 subset order):
      sz.num_simple_radial,                                    // r=0
      sz.num_simple_radial_fixed_pose,                         // r=1 {pose}
      sz.num_simple_radial_fixed_focal,                        // r=1 {focal}
      sz.num_simple_radial_fixed_extra_calib,                  // r=1 {extra_calib}
      sz.num_simple_radial_fixed_point,                        // r=1 {point}
      sz.num_simple_radial_fixed_pose_fixed_focal,             // r=2 {pose,focal}
      sz.num_simple_radial_fixed_pose_fixed_extra_calib,       // r=2 {pose,extra_calib}
      sz.num_simple_radial_fixed_pose_fixed_point,             // r=2 {pose,point}
      sz.num_simple_radial_fixed_focal_fixed_extra_calib,      // r=2 {focal,extra_calib} reserved
      sz.num_simple_radial_fixed_focal_fixed_point,            // r=2 {focal,point}
      sz.num_simple_radial_fixed_extra_calib_fixed_point,      // r=2 {extra_calib,point}
      sz.num_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib,    // r=3 reserved
      sz.num_simple_radial_fixed_pose_fixed_focal_fixed_point,          // r=3
      sz.num_simple_radial_fixed_pose_fixed_extra_calib_fixed_point,    // r=3
      sz.num_simple_radial_fixed_focal_fixed_extra_calib_fixed_point,   // r=3 reserved
      // pinhole factor counts (same subset order):
      sz.num_pinhole,                                          // r=0
      sz.num_pinhole_fixed_pose,                               // r=1 {pose}
      sz.num_pinhole_fixed_focal,                              // r=1 {focal}
      sz.num_pinhole_fixed_extra_calib,                        // r=1 {extra_calib}
      sz.num_pinhole_fixed_point,                              // r=1 {point}
      sz.num_pinhole_fixed_pose_fixed_focal,                   // r=2 {pose,focal}
      sz.num_pinhole_fixed_pose_fixed_extra_calib,             // r=2 {pose,extra_calib}
      sz.num_pinhole_fixed_pose_fixed_point,                   // r=2 {pose,point}
      sz.num_pinhole_fixed_focal_fixed_extra_calib,            // r=2 {focal,extra_calib} reserved
      sz.num_pinhole_fixed_focal_fixed_point,                  // r=2 {focal,point}
      sz.num_pinhole_fixed_extra_calib_fixed_point,            // r=2 {extra_calib,point}
      sz.num_pinhole_fixed_pose_fixed_focal_fixed_extra_calib, // r=3 reserved
      sz.num_pinhole_fixed_pose_fixed_focal_fixed_point,       // r=3
      sz.num_pinhole_fixed_pose_fixed_extra_calib_fixed_point, // r=3
      sz.num_pinhole_fixed_focal_fixed_extra_calib_fixed_point // r=3 reserved
  );
}

}  // namespace colmap
