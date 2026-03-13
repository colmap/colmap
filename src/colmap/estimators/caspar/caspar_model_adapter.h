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

  // SimpleRadial — all 7 variants generated, 4 dispatched
  size_t num_simple_radial_calibs = 0;
  size_t num_simple_radial = 0;
  size_t num_simple_radial_fixed_pose = 0;
  size_t num_simple_radial_fixed_point = 0;
  size_t num_simple_radial_fixed_calib = 0;  // reserved
  size_t num_simple_radial_fixed_pose_fixed_point = 0;
  size_t num_simple_radial_fixed_pose_fixed_calib = 0;   // reserved
  size_t num_simple_radial_fixed_point_fixed_calib = 0;  // reserved

  // Pinhole
  size_t num_pinhole_calibs = 0;
  size_t num_pinhole = 0;
  size_t num_pinhole_fixed_pose = 0;
  size_t num_pinhole_fixed_point = 0;
  size_t num_pinhole_fixed_calib = 0;  // reserved
  size_t num_pinhole_fixed_pose_fixed_point = 0;
  size_t num_pinhole_fixed_pose_fixed_calib = 0;   // reserved
  size_t num_pinhole_fixed_point_fixed_calib = 0;  // reserved
};

// Interface — one implementation per camera model.

class ICasparModelAdapter {
 public:
  virtual ~ICasparModelAdapter() = default;

  virtual CameraModelId ModelId() const = 0;
  virtual size_t CalibSize() const = 0;

  virtual void FillSizing(CasparSolverSizing& sz,
                          const ModelData& md,
                          size_t num_calibs) const = 0;

  virtual void ExtractCalib(const Camera& camera,
                            std::vector<StorageType>& out) const = 0;

  virtual void WriteCalib(Camera& camera,
                          const StorageType* data,
                          size_t idx) const = 0;

  virtual void SetCalibNodes(caspar::GraphSolver& solver,
                             StorageType* data,
                             size_t n) const = 0;

  virtual void GetCalibNodes(caspar::GraphSolver& solver,
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
  size_t CalibSize() const override { return 4; }  // [f, cx, cy, k]

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_simple_radial_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      auto variant = static_cast<FactorVariant>(v);
      switch (variant) {
        case FactorVariant::BASE:
          sz.num_simple_radial = md.variants[v].num_factors;
        case FactorVariant::FIXED_POINT:
          sz.num_simple_radial_fixed_point = md.variants[v].num_factors;
        case FactorVariant::FIXED_POSE:
          sz.num_simple_radial_fixed_pose = md.variants[v].num_factors;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_simple_radial_fixed_pose_fixed_point =
              md.variants[v].num_factors;
      }
    }
  }

  void ExtractCalib(const Camera& camera,
                    std::vector<StorageType>& out) const override {
    for (const double p : camera.params)
      out.push_back(static_cast<StorageType>(p));
  }

  void WriteCalib(Camera& camera,
                  const StorageType* data,
                  size_t idx) const override {
    for (size_t i = 0; i < camera.params.size(); ++i)
      camera.params[i] = static_cast<double>(data[idx * CalibSize() + i]);
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

  virtual void SetVariantFactors(caspar::GraphSolver& s,
                                 FactorVariant variant,
                                 const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      case FactorVariant::BASE:
        s.set_simple_radial_pose_indices_from_host(d.pose_indices.data(), n);
        s.set_simple_radial_calib_indices_from_host(d.calib_indices.data(), n);
        s.set_simple_radial_point_indices_from_host(d.point_indices.data(), n);
        s.set_simple_radial_pixel_data_from_stacked_host(d.pixels.data(), 0, n);
        s.set_simple_radial_num(n);
      case FactorVariant::FIXED_POINT:
        s.set_simple_radial_fixed_point_pose_indices_from_host(
            d.pose_indices.data(), n);
        s.set_simple_radial_fixed_point_calib_indices_from_host(
            d.calib_indices.data(), n);
        s.set_simple_radial_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_simple_radial_fixed_point_num(n);
      case FactorVariant::FIXED_POSE:
        s.set_simple_radial_fixed_pose_calib_indices_from_host(
            d.calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_point_indices_from_host(
            d.point_indices.data(), n);
        s.set_simple_radial_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_simple_radial_fixed_pose_num(n);
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_simple_radial_fixed_pose_fixed_point_calib_indices_from_host(
            d.calib_indices.data(), n);
        s.set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_simple_radial_fixed_pose_fixed_point_num(n);
    }
  }
};

// Pinhole implementation

class PinholeAdapter : public ICasparModelAdapter {
 public:
  CameraModelId ModelId() const override { return CameraModelId::kPinhole; }
  size_t CalibSize() const override { return 4; }  // [fx, fy, cx, cy]

  void FillSizing(CasparSolverSizing& sz,
                  const ModelData& md,
                  size_t num_calibs) const override {
    sz.num_pinhole_calibs = num_calibs;
    for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
      auto variant = static_cast<FactorVariant>(v);
      switch (variant) {
        case FactorVariant::BASE:
          sz.num_pinhole = md.variants[v].num_factors;
        case FactorVariant::FIXED_POINT:
          sz.num_pinhole_fixed_point = md.variants[v].num_factors;
        case FactorVariant::FIXED_POSE:
          sz.num_pinhole_fixed_pose = md.variants[v].num_factors;
        case FactorVariant::FIXED_POSE_FIXED_POINT:
          sz.num_pinhole_fixed_pose_fixed_point = md.variants[v].num_factors;
      }
    }
  }

  void ExtractCalib(const Camera& camera,
                    std::vector<StorageType>& out) const override {
    for (const double p : camera.params)
      out.push_back(static_cast<StorageType>(p));
  }

  void WriteCalib(Camera& camera,
                  const StorageType* data,
                  size_t idx) const override {
    for (size_t i = 0; i < camera.params.size(); ++i)
      camera.params[i] = static_cast<double>(data[idx * CalibSize() + i]);
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
  virtual void SetVariantFactors(caspar::GraphSolver& s,
                                 FactorVariant variant,
                                 const VariantData& d) const override {
    const size_t n = d.num_factors;
    switch (variant) {
      case FactorVariant::BASE:
        s.set_pinhole_pose_indices_from_host(d.pose_indices.data(), n);
        s.set_pinhole_calib_indices_from_host(d.calib_indices.data(), n);
        s.set_pinhole_point_indices_from_host(d.point_indices.data(), n);
        s.set_pinhole_pixel_data_from_stacked_host(d.pixels.data(), 0, n);
        s.set_pinhole_num(n);
      case FactorVariant::FIXED_POINT:
        s.set_pinhole_fixed_point_pose_indices_from_host(d.pose_indices.data(),
                                                         n);
        s.set_pinhole_fixed_point_calib_indices_from_host(
            d.calib_indices.data(), n);
        s.set_pinhole_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_pinhole_fixed_point_num(n);
      case FactorVariant::FIXED_POSE:
        s.set_pinhole_fixed_pose_calib_indices_from_host(d.calib_indices.data(),
                                                         n);
        s.set_pinhole_fixed_pose_point_indices_from_host(d.point_indices.data(),
                                                         n);
        s.set_pinhole_fixed_pose_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_pinhole_fixed_pose_num(n);
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        s.set_pinhole_fixed_pose_fixed_point_calib_indices_from_host(
            d.calib_indices.data(), n);
        s.set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host(
            d.const_poses.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host(
            d.const_points.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
            d.pixels.data(), 0, n);
        s.set_pinhole_fixed_pose_fixed_point_num(n);
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
inline caspar::GraphSolver CreateSolver(
    const caspar::SolverParams<StorageType>& params,
    const CasparSolverSizing& sz) {
  // WARNING: This call is very tedious and bug-prone, and will change in a
  // newer release of Caspar
  return caspar::GraphSolver(params,
                             sz.num_pinhole_calibs,
                             sz.num_points,
                             sz.num_poses,
                             sz.num_simple_radial_calibs,
                             sz.num_simple_radial,
                             sz.num_simple_radial_fixed_pose,
                             sz.num_simple_radial_fixed_point,
                             sz.num_simple_radial_fixed_calib,
                             sz.num_simple_radial_fixed_pose_fixed_point,
                             sz.num_simple_radial_fixed_pose_fixed_calib,
                             sz.num_pinhole_fixed_point_fixed_calib,
                             sz.num_pinhole,
                             sz.num_pinhole_fixed_pose,
                             sz.num_pinhole_fixed_point,
                             sz.num_simple_radial_fixed_calib,
                             sz.num_simple_radial_fixed_pose_fixed_point,
                             sz.num_simple_radial_fixed_pose_fixed_calib,
                             sz.num_simple_radial_fixed_point_fixed_calib);
}

}  // namespace colmap