// Update this file if the Caspar solver API changes for this model.
#pragma once

#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"

#include <vector>

#include <solver.h>

#ifdef CASPAR_USE_DOUBLE
typedef double StorageType;
#else
typedef float StorageType;
#endif

namespace colmap {
class CasparModelAdapter {
 public:
  static constexpr CameraModelId kModelId = CameraModelId::kPinhole;
  static constexpr size_t kCalibSize = 4;  // fx, fy, cx, cy

  static bool IsSupported(const CameraModelId id) { return id == kModelId; }

  static void ExtractCalib(const Camera& camera,
                           std::vector<StorageType>& out) {
    for (const double p : camera.params)
      out.push_back(static_cast<StorageType>(p));
  }

  // idx is the camera's index in the flat buffer, not byte offset
  static void WriteCalib(Camera& camera,
                         const StorageType* data,
                         const size_t idx) {
    for (size_t i = 0; i < camera.params.size(); ++i) {
      camera.params[i] = data[idx * kCalibSize + i];
    }
  }

  static void SetCalibNodes(caspar::GraphSolver& s,
                            StorageType* data,
                            const size_t n) {
    s.set_PinholeCalib_nodes_from_stacked_host(data, 0, n);
  }

  static void GetCalibNodes(caspar::GraphSolver& s,
                            StorageType* data,
                            const size_t n) {
    s.get_PinholeCalib_nodes_to_stacked_host(data, 0, n);
  }

  static void SetFactorIndices(caspar::GraphSolver& s,
                               const unsigned int* pose_idx,
                               const unsigned int* calib_idx,
                               const unsigned int* point_idx,
                               const StorageType* pixels,
                               const size_t n) {
    s.set_pinhole_pose_indices_from_host(pose_idx, n);
    s.set_pinhole_calib_indices_from_host(calib_idx, n);
    s.set_pinhole_point_indices_from_host(point_idx, n);
    s.set_pinhole_pixel_data_from_stacked_host(pixels, 0, n);
    s.set_pinhole_num(n);
  }

  static void SetFixedPoseFactorIndices(caspar::GraphSolver& s,
                                        const unsigned int* calib_idx,
                                        const unsigned int* point_idx,
                                        const StorageType* poses,
                                        const StorageType* pixels,
                                        const size_t n) {
    s.set_pinhole_fixed_pose_calib_indices_from_host(calib_idx, n);
    s.set_pinhole_fixed_pose_point_indices_from_host(point_idx, n);
    s.set_pinhole_fixed_pose_cam_T_world_data_from_stacked_host(poses, 0, n);
    s.set_pinhole_fixed_pose_pixel_data_from_stacked_host(pixels, 0, n);
    s.set_pinhole_fixed_pose_num(n);
  }

  static void SetFixedPointFactorIndices(caspar::GraphSolver& s,
                                         const unsigned int* pose_idx,
                                         const unsigned int* calib_idx,
                                         const StorageType* points,
                                         const StorageType* pixels,
                                         const size_t n) {
    s.set_pinhole_fixed_point_pose_indices_from_host(pose_idx, n);
    s.set_pinhole_fixed_point_calib_indices_from_host(calib_idx, n);
    s.set_pinhole_fixed_point_point_data_from_stacked_host(points, 0, n);
    s.set_pinhole_fixed_point_pixel_data_from_stacked_host(pixels, 0, n);
    s.set_pinhole_fixed_point_num(n);
  }
};

static caspar::GraphSolver CreateSolver(
    const caspar::SolverParams<StorageType>& params,
    const size_t num_calibs,
    const size_t num_points,
    const size_t num_poses,
    const size_t num_factors,
    const size_t num_fixed_pose,
    const size_t num_fixed_point) {
  return caspar::GraphSolver(params,
                             num_calibs,
                             num_points,
                             num_poses,
                             num_factors,
                             num_fixed_pose,
                             num_fixed_point);
}

}  // namespace colmap