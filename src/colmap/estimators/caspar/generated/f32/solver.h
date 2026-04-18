#pragma once

#include <cstdint>
#include <vector>

#include "shared_indices.h"
#include "solver_params.h"
#include <cuda_runtime.h>

namespace caspar {

enum class ExitReason {
  MAX_ITERATIONS,
  CONVERGED_SCORE_THRESHOLD,
  CONVERGED_DIAG_EXIT
};

struct IterationData {
  int solver_iter;
  int pcg_iter;
  double score_current;
  double score_best;
  double step_quality;
  double diag;
  double dt_inc;
  double dt_tot;
  bool step_accepted;
};

struct SolveResult {
  double initial_score;
  double final_score;
  int iteration_count;
  double runtime;
  ExitReason exit_reason;
  std::vector<IterationData> iterations;
};

class GraphSolver {
 public:
  /**
   * Base constructor.
   *
   * @param params: The params to use for the solver
   * @param PinholeExtraCalib_num_max the maximum number of PinholeExtraCalibs
   * @param PinholeFocal_num_max the maximum number of PinholeFocals
   * @param Point_num_max the maximum number of Points
   * @param Pose_num_max the maximum number of Poses
   * @param SimpleRadialExtraCalib_num_max the maximum number of
   * SimpleRadialExtraCalibs
   * @param SimpleRadialFocal_num_max the maximum number of SimpleRadialFocals
   * @param simple_radial_num_max the maximum number of simple_radials
   * @param simple_radial_fixed_pose_num_max the maximum number of
   * simple_radial_fixed_poses
   * @param simple_radial_fixed_focal_num_max the maximum number of
   * simple_radial_fixed_focals
   * @param simple_radial_fixed_extra_calib_num_max the maximum number of
   * simple_radial_fixed_extra_calibs
   * @param simple_radial_fixed_point_num_max the maximum number of
   * simple_radial_fixed_points
   * @param simple_radial_fixed_pose_fixed_focal_num_max the maximum number of
   * simple_radial_fixed_pose_fixed_focals
   * @param simple_radial_fixed_pose_fixed_extra_calib_num_max the maximum
   * number of simple_radial_fixed_pose_fixed_extra_calibs
   * @param simple_radial_fixed_pose_fixed_point_num_max the maximum number of
   * simple_radial_fixed_pose_fixed_points
   * @param simple_radial_fixed_focal_fixed_extra_calib_num_max the maximum
   * number of simple_radial_fixed_focal_fixed_extra_calibs
   * @param simple_radial_fixed_focal_fixed_point_num_max the maximum number of
   * simple_radial_fixed_focal_fixed_points
   * @param simple_radial_fixed_extra_calib_fixed_point_num_max the maximum
   * number of simple_radial_fixed_extra_calib_fixed_points
   * @param simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max the
   * maximum number of simple_radial_fixed_pose_fixed_focal_fixed_extra_calibs
   * @param simple_radial_fixed_pose_fixed_focal_fixed_point_num_max the maximum
   * number of simple_radial_fixed_pose_fixed_focal_fixed_points
   * @param simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max the
   * maximum number of simple_radial_fixed_pose_fixed_extra_calib_fixed_points
   * @param simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max the
   * maximum number of simple_radial_fixed_focal_fixed_extra_calib_fixed_points
   * @param pinhole_num_max the maximum number of pinholes
   * @param pinhole_fixed_pose_num_max the maximum number of pinhole_fixed_poses
   * @param pinhole_fixed_focal_num_max the maximum number of
   * pinhole_fixed_focals
   * @param pinhole_fixed_extra_calib_num_max the maximum number of
   * pinhole_fixed_extra_calibs
   * @param pinhole_fixed_point_num_max the maximum number of
   * pinhole_fixed_points
   * @param pinhole_fixed_pose_fixed_focal_num_max the maximum number of
   * pinhole_fixed_pose_fixed_focals
   * @param pinhole_fixed_pose_fixed_extra_calib_num_max the maximum number of
   * pinhole_fixed_pose_fixed_extra_calibs
   * @param pinhole_fixed_pose_fixed_point_num_max the maximum number of
   * pinhole_fixed_pose_fixed_points
   * @param pinhole_fixed_focal_fixed_extra_calib_num_max the maximum number of
   * pinhole_fixed_focal_fixed_extra_calibs
   * @param pinhole_fixed_focal_fixed_point_num_max the maximum number of
   * pinhole_fixed_focal_fixed_points
   * @param pinhole_fixed_extra_calib_fixed_point_num_max the maximum number of
   * pinhole_fixed_extra_calib_fixed_points
   * @param pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max the maximum
   * number of pinhole_fixed_pose_fixed_focal_fixed_extra_calibs
   * @param pinhole_fixed_pose_fixed_focal_fixed_point_num_max the maximum
   * number of pinhole_fixed_pose_fixed_focal_fixed_points
   * @param pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max the maximum
   * number of pinhole_fixed_pose_fixed_extra_calib_fixed_points
   * @param pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max the
   * maximum number of pinhole_fixed_focal_fixed_extra_calib_fixed_points
   */
  GraphSolver(
      const SolverParams<double>& params,
      size_t PinholeExtraCalib_num_max,
      size_t PinholeFocal_num_max,
      size_t Point_num_max,
      size_t Pose_num_max,
      size_t SimpleRadialExtraCalib_num_max,
      size_t SimpleRadialFocal_num_max,
      size_t simple_radial_num_max,
      size_t simple_radial_fixed_pose_num_max,
      size_t simple_radial_fixed_focal_num_max,
      size_t simple_radial_fixed_extra_calib_num_max,
      size_t simple_radial_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_focal_num_max,
      size_t simple_radial_fixed_pose_fixed_extra_calib_num_max,
      size_t simple_radial_fixed_pose_fixed_point_num_max,
      size_t simple_radial_fixed_focal_fixed_extra_calib_num_max,
      size_t simple_radial_fixed_focal_fixed_point_num_max,
      size_t simple_radial_fixed_extra_calib_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max,
      size_t simple_radial_fixed_pose_fixed_focal_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max,
      size_t simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max,
      size_t pinhole_num_max,
      size_t pinhole_fixed_pose_num_max,
      size_t pinhole_fixed_focal_num_max,
      size_t pinhole_fixed_extra_calib_num_max,
      size_t pinhole_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_focal_num_max,
      size_t pinhole_fixed_pose_fixed_extra_calib_num_max,
      size_t pinhole_fixed_pose_fixed_point_num_max,
      size_t pinhole_fixed_focal_fixed_extra_calib_num_max,
      size_t pinhole_fixed_focal_fixed_point_num_max,
      size_t pinhole_fixed_extra_calib_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max,
      size_t pinhole_fixed_pose_fixed_focal_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max,
      size_t pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max);

  // This class is managing cuda memory and cannot be copied.
  GraphSolver(const GraphSolver&) = delete;
  GraphSolver& operator=(const GraphSolver&) = delete;

  GraphSolver(GraphSolver&&) = default;
  GraphSolver& operator=(GraphSolver&&) = default;

  ~GraphSolver();

  /**
   * Set the solver parameters.
   */
  void set_params(const SolverParams<double>& params);

  /**
   * Run the solver.
   */
  SolveResult solve(bool print_progress = false, bool verbose_logging = false);

  /**
   * Finish the indices.
   *
   * This function has to be called after all indices are set and before the
   * solve function is called.
   */
  void finish_indices();

  /**
   * Get the number of allocated bytes.
   */
  size_t get_allocation_size();

  /**
   * Set the current value for the PinholeExtraCalib nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_PinholeExtraCalib_nodes_from_stacked_host(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the current value for the PinholeExtraCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_PinholeExtraCalib_nodes_from_stacked_device(const float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Read the current value for the PinholeExtraCalib nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_PinholeExtraCalib_nodes_to_stacked_host(float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Read the current value for the PinholeExtraCalib nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_PinholeExtraCalib_nodes_to_stacked_device(float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the current number of active nodes of type PinholeExtraCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_PinholeExtraCalib_num(size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_PinholeFocal_nodes_from_stacked_host(const float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_PinholeFocal_nodes_from_stacked_device(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_PinholeFocal_nodes_to_stacked_host(float* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_PinholeFocal_nodes_to_stacked_device(float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current number of active nodes of type PinholeFocal.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_PinholeFocal_num(size_t num);

  /**
   * Set the current value for the Point nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_Point_nodes_from_stacked_host(const float* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Set the current value for the Point nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_Point_nodes_from_stacked_device(const float* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_Point_nodes_to_stacked_host(float* const data,
                                       size_t offset,
                                       size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_Point_nodes_to_stacked_device(float* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Set the current number of active nodes of type Point.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_Point_num(size_t num);

  /**
   * Set the current value for the Pose nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_Pose_nodes_from_stacked_host(const float* const data,
                                        size_t offset,
                                        size_t num);

  /**
   * Set the current value for the Pose nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_Pose_nodes_from_stacked_device(const float* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Read the current value for the Pose nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_Pose_nodes_to_stacked_host(float* const data,
                                      size_t offset,
                                      size_t num);

  /**
   * Read the current value for the Pose nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_Pose_nodes_to_stacked_device(float* const data,
                                        size_t offset,
                                        size_t num);

  /**
   * Set the current number of active nodes of type Pose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_Pose_num(size_t num);

  /**
   * Set the current value for the SimpleRadialExtraCalib nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_SimpleRadialExtraCalib_nodes_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialExtraCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_SimpleRadialExtraCalib_nodes_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialExtraCalib nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_SimpleRadialExtraCalib_nodes_to_stacked_host(float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Read the current value for the SimpleRadialExtraCalib nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_SimpleRadialExtraCalib_nodes_to_stacked_device(float* const data,
                                                          size_t offset,
                                                          size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialExtraCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_SimpleRadialExtraCalib_num(size_t num);

  /**
   * Set the current value for the SimpleRadialFocal nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_SimpleRadialFocal_nodes_from_stacked_host(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the current value for the SimpleRadialFocal nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void set_SimpleRadialFocal_nodes_from_stacked_device(const float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Read the current value for the SimpleRadialFocal nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_SimpleRadialFocal_nodes_to_stacked_host(float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Read the current value for the SimpleRadialFocal nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void get_SimpleRadialFocal_nodes_to_stacked_device(float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialFocal.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_SimpleRadialFocal_num(size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial factor from
   * host.
   */
  void set_simple_radial_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial factor from
   * device.
   */
  void set_simple_radial_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial factor from
   * host.
   */
  void set_simple_radial_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial factor from
   * device.
   */
  void set_simple_radial_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the simple_radial factor
   * from host.
   */
  void set_simple_radial_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the simple_radial factor
   * from device.
   */
  void set_simple_radial_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial factor from
   * host.
   */
  void set_simple_radial_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial factor from
   * device.
   */
  void set_simple_radial_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_pixel_data_from_stacked_host(const float* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the values for the pixel consts simple_radial factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_pixel_data_from_stacked_device(const float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Set the current number of simple_radial factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_num(size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial_fixed_pose
   * factor from host.
   */
  void set_simple_radial_fixed_pose_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial_fixed_pose
   * factor from device.
   */
  void set_simple_radial_fixed_pose_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose factor from host.
   */
  void set_simple_radial_fixed_pose_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose factor from device.
   */
  void set_simple_radial_fixed_pose_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial_fixed_pose
   * factor from host.
   */
  void set_simple_radial_fixed_pose_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial_fixed_pose
   * factor from device.
   */
  void set_simple_radial_fixed_pose_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_pose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_num(size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial_fixed_focal
   * factor from host.
   */
  void set_simple_radial_fixed_focal_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial_fixed_focal
   * factor from device.
   */
  void set_simple_radial_fixed_focal_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_focal factor from host.
   */
  void set_simple_radial_fixed_focal_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_focal factor from device.
   */
  void set_simple_radial_fixed_focal_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial_fixed_focal
   * factor from host.
   */
  void set_simple_radial_fixed_focal_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the simple_radial_fixed_focal
   * factor from device.
   */
  void set_simple_radial_fixed_focal_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_focal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_focal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_focal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_focal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_focal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_focal_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_extra_calib_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_extra_calib_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_extra_calib_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_extra_calib_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_extra_calib factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_extra_calib factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts simple_radial_fixed_extra_calib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts simple_radial_fixed_extra_calib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_extra_calib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial_fixed_point
   * factor from host.
   */
  void set_simple_radial_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the simple_radial_fixed_point
   * factor from device.
   */
  void set_simple_radial_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial_fixed_point
   * factor from host.
   */
  void set_simple_radial_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the simple_radial_fixed_point
   * factor from device.
   */
  void set_simple_radial_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_point factor from host.
   */
  void set_simple_radial_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_point factor from device.
   */
  void set_simple_radial_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_point factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_point factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_point factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_point factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_point_num(size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_focal factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_focal factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_focal factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_focal_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_focal factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_focal_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_pose_fixed_focal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_pose_fixed_focal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_focal_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_pose_fixed_extra_calib
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_point factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_point factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_point factor from host.
   */
  void set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_point factor from device.
   */
  void set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_pose_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts simple_radial_fixed_pose_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_pose_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_pose_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_pose_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_point_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_extra_calib factor from device.
   */
  void set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_focal_fixed_extra_calib factor from host.
   */
  void set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_focal_fixed_extra_calib factor from device.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_focal_fixed_extra_calib
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_focal_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_point factor from host.
   */
  void set_simple_radial_fixed_focal_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_point factor from device.
   */
  void set_simple_radial_fixed_focal_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_focal_fixed_point factor from host.
   */
  void set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_focal_fixed_point factor from device.
   */
  void
  set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_focal_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts simple_radial_fixed_focal_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_focal_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts simple_radial_fixed_focal_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_focal_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts simple_radial_fixed_focal_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_focal_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_focal_fixed_point_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_extra_calib_fixed_point factor from host.
   */
  void set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_extra_calib_fixed_point factor from device.
   */
  void set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_extra_calib_fixed_point factor from host.
   */
  void set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_extra_calib_fixed_point factor from device.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_extra_calib_fixed_point
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_extra_calib_fixed_point_num(size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from host.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from device.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num(
      size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from host.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from device.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of simple_radial_fixed_pose_fixed_focal_fixed_point
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_focal_fixed_point_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from host.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from device.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * simple_radial_fixed_pose_fixed_extra_calib_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from host.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from device.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * simple_radial_fixed_focal_fixed_extra_calib_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num(
      size_t num);

  /**
   * Set the indices for the pose argument for the pinhole factor from host.
   */
  void set_pinhole_pose_indices_from_host(const unsigned int* const indices,
                                          size_t num);

  /**
   * Set the indices for the pose argument for the pinhole factor from device.
   */
  void set_pinhole_pose_indices_from_device(const unsigned int* const indices,
                                            size_t num);

  /**
   * Set the indices for the focal argument for the pinhole factor from host.
   */
  void set_pinhole_focal_indices_from_host(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the focal argument for the pinhole factor from device.
   */
  void set_pinhole_focal_indices_from_device(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole factor from
   * host.
   */
  void set_pinhole_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole factor from
   * device.
   */
  void set_pinhole_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole factor from host.
   */
  void set_pinhole_point_indices_from_host(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the point argument for the pinhole factor from device.
   */
  void set_pinhole_point_indices_from_device(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the values for the pixel consts pinhole factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_pixel_data_from_stacked_host(const float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the values for the pixel consts pinhole factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_pixel_data_from_stacked_device(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the current number of pinhole factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_num(size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_pose factor
   * from host.
   */
  void set_pinhole_fixed_pose_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_pose factor
   * from device.
   */
  void set_pinhole_fixed_pose_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_pose
   * factor from host.
   */
  void set_pinhole_fixed_pose_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_pose
   * factor from device.
   */
  void set_pinhole_fixed_pose_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_pose factor
   * from host.
   */
  void set_pinhole_fixed_pose_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_pose factor
   * from device.
   */
  void set_pinhole_fixed_pose_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_num(size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_focal factor
   * from host.
   */
  void set_pinhole_fixed_focal_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_focal factor
   * from device.
   */
  void set_pinhole_fixed_focal_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_focal
   * factor from host.
   */
  void set_pinhole_fixed_focal_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_focal
   * factor from device.
   */
  void set_pinhole_fixed_focal_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_focal factor
   * from host.
   */
  void set_pinhole_fixed_focal_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_focal factor
   * from device.
   */
  void set_pinhole_fixed_focal_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_focal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_focal_num(size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_extra_calib
   * factor from host.
   */
  void set_pinhole_fixed_extra_calib_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_extra_calib
   * factor from device.
   */
  void set_pinhole_fixed_extra_calib_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_extra_calib
   * factor from host.
   */
  void set_pinhole_fixed_extra_calib_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_extra_calib
   * factor from device.
   */
  void set_pinhole_fixed_extra_calib_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_extra_calib
   * factor from host.
   */
  void set_pinhole_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the pinhole_fixed_extra_calib
   * factor from device.
   */
  void set_pinhole_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_extra_calib factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_extra_calib factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts pinhole_fixed_extra_calib factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts pinhole_fixed_extra_calib factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_extra_calib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_point factor
   * from host.
   */
  void set_pinhole_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the pinhole_fixed_point factor
   * from device.
   */
  void set_pinhole_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_point factor
   * from host.
   */
  void set_pinhole_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the pinhole_fixed_point factor
   * from device.
   */
  void set_pinhole_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_point
   * factor from host.
   */
  void set_pinhole_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the pinhole_fixed_point
   * factor from device.
   */
  void set_pinhole_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_point factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_point_num(size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_focal factor from host.
   */
  void set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_focal factor from device.
   */
  void set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_focal factor from host.
   */
  void set_pinhole_fixed_pose_fixed_focal_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_focal factor from device.
   */
  void set_pinhole_fixed_pose_fixed_focal_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_focal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_focal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_focal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_focal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_pose_fixed_focal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_pose_fixed_focal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_focal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_focal_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_extra_calib factor from host.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_extra_calib factor from device.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_extra_calib factor from host.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_extra_calib factor from device.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_extra_calib factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_extra_calib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_point factor from host.
   */
  void set_pinhole_fixed_pose_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_point factor from device.
   */
  void set_pinhole_fixed_pose_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_point factor from host.
   */
  void set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_point factor from device.
   */
  void set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_pose_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts pinhole_fixed_pose_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_pose_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_pose_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_point_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_extra_calib factor from host.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_extra_calib factor from device.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_focal_fixed_extra_calib factor from host.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_focal_fixed_extra_calib factor from device.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_focal_fixed_extra_calib factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_focal_fixed_extra_calib factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_focal_fixed_extra_calib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_point factor from host.
   */
  void set_pinhole_fixed_focal_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_point factor from device.
   */
  void set_pinhole_fixed_focal_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_focal_fixed_point factor from host.
   */
  void set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_focal_fixed_point factor from device.
   */
  void set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_focal_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts pinhole_fixed_focal_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_focal_fixed_point factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_focal_fixed_point factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_focal_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_focal_fixed_point_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_extra_calib_fixed_point factor from host.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_extra_calib_fixed_point factor from device.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_extra_calib_fixed_point factor from host.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_extra_calib_fixed_point factor from device.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_extra_calib_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts pinhole_fixed_extra_calib_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_extra_calib_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_extra_calib_fixed_point factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_extra_calib_fixed_point
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts pinhole_fixed_extra_calib_fixed_point
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_extra_calib_fixed_point factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_extra_calib_fixed_point_num(size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from host.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from device.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_focal_fixed_extra_calib
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num(size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from host.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the extra_calib argument for the
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from device.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_focal_fixed_point
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_focal_fixed_point_num(size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from host.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from device.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_pose_fixed_extra_calib_fixed_point
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num(size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from host.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from device.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the extra_calib consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of pinhole_fixed_focal_fixed_extra_calib_fixed_point
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_num(size_t num);

 private:
  SolverParams<float> params_;
  uint8_t* origin_ptr_;
  size_t scratch_inout_size_;
  size_t allocation_size_;

  int solver_iter_;
  int pcg_iter_;

  bool indices_valid_;

  float pcg_r_0_norm2_;
  float pcg_r_kp1_norm2_;

  size_t PinholeExtraCalib_num_;
  size_t PinholeExtraCalib_num_max_;
  size_t PinholeFocal_num_;
  size_t PinholeFocal_num_max_;
  size_t Point_num_;
  size_t Point_num_max_;
  size_t Pose_num_;
  size_t Pose_num_max_;
  size_t SimpleRadialExtraCalib_num_;
  size_t SimpleRadialExtraCalib_num_max_;
  size_t SimpleRadialFocal_num_;
  size_t SimpleRadialFocal_num_max_;
  size_t simple_radial_num_;
  size_t simple_radial_num_max_;
  size_t simple_radial_fixed_pose_num_;
  size_t simple_radial_fixed_pose_num_max_;
  size_t simple_radial_fixed_focal_num_;
  size_t simple_radial_fixed_focal_num_max_;
  size_t simple_radial_fixed_extra_calib_num_;
  size_t simple_radial_fixed_extra_calib_num_max_;
  size_t simple_radial_fixed_point_num_;
  size_t simple_radial_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_focal_num_;
  size_t simple_radial_fixed_pose_fixed_focal_num_max_;
  size_t simple_radial_fixed_pose_fixed_extra_calib_num_;
  size_t simple_radial_fixed_pose_fixed_extra_calib_num_max_;
  size_t simple_radial_fixed_pose_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_point_num_max_;
  size_t simple_radial_fixed_focal_fixed_extra_calib_num_;
  size_t simple_radial_fixed_focal_fixed_extra_calib_num_max_;
  size_t simple_radial_fixed_focal_fixed_point_num_;
  size_t simple_radial_fixed_focal_fixed_point_num_max_;
  size_t simple_radial_fixed_extra_calib_fixed_point_num_;
  size_t simple_radial_fixed_extra_calib_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_;
  size_t simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max_;
  size_t simple_radial_fixed_pose_fixed_focal_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_focal_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max_;
  size_t simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_;
  size_t simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max_;
  size_t pinhole_num_;
  size_t pinhole_num_max_;
  size_t pinhole_fixed_pose_num_;
  size_t pinhole_fixed_pose_num_max_;
  size_t pinhole_fixed_focal_num_;
  size_t pinhole_fixed_focal_num_max_;
  size_t pinhole_fixed_extra_calib_num_;
  size_t pinhole_fixed_extra_calib_num_max_;
  size_t pinhole_fixed_point_num_;
  size_t pinhole_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_num_;
  size_t pinhole_fixed_pose_fixed_focal_num_max_;
  size_t pinhole_fixed_pose_fixed_extra_calib_num_;
  size_t pinhole_fixed_pose_fixed_extra_calib_num_max_;
  size_t pinhole_fixed_pose_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_point_num_max_;
  size_t pinhole_fixed_focal_fixed_extra_calib_num_;
  size_t pinhole_fixed_focal_fixed_extra_calib_num_max_;
  size_t pinhole_fixed_focal_fixed_point_num_;
  size_t pinhole_fixed_focal_fixed_point_num_max_;
  size_t pinhole_fixed_extra_calib_fixed_point_num_;
  size_t pinhole_fixed_extra_calib_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_;
  size_t pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_focal_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max_;
  size_t pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_;
  size_t pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max_;

  size_t get_nbytes();
  float linearize_first();
  void linearize();
  float do_res_jac_first();
  void do_res_jac();
  void do_normalize();
  void do_jtjp_direct();
  void do_alpha_first();
  void do_alpha();
  void do_update_step_first();
  void do_update_step();
  void do_update_r_first();
  void do_update_r();
  float do_retract_score();
  void do_beta();
  void do_update_p();
  void do_update_Mp();
  float get_pred_decrease();

  float* marker__start_;
  float* nodes__PinholeExtraCalib__storage_current_;
  float* nodes__PinholeExtraCalib__storage_check_;
  float* nodes__PinholeExtraCalib__storage_new_best_;
  float* nodes__PinholeFocal__storage_current_;
  float* nodes__PinholeFocal__storage_check_;
  float* nodes__PinholeFocal__storage_new_best_;
  float* nodes__Point__storage_current_;
  float* nodes__Point__storage_check_;
  float* nodes__Point__storage_new_best_;
  float* nodes__Pose__storage_current_;
  float* nodes__Pose__storage_check_;
  float* nodes__Pose__storage_new_best_;
  float* nodes__SimpleRadialExtraCalib__storage_current_;
  float* nodes__SimpleRadialExtraCalib__storage_check_;
  float* nodes__SimpleRadialExtraCalib__storage_new_best_;
  float* nodes__SimpleRadialFocal__storage_current_;
  float* nodes__SimpleRadialFocal__storage_check_;
  float* nodes__SimpleRadialFocal__storage_new_best_;
  SharedIndex* facs__simple_radial__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial__args__focal__idx_shared_;
  SharedIndex* facs__simple_radial__args__extra_calib__idx_shared_;
  SharedIndex* facs__simple_radial__args__point__idx_shared_;
  float* facs__simple_radial__args__pixel__data_;
  SharedIndex* facs__simple_radial_fixed_pose__args__focal__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_pose__args__extra_calib__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_pose__args__point__idx_shared_;
  float* facs__simple_radial_fixed_pose__args__pixel__data_;
  float* facs__simple_radial_fixed_pose__args__pose__data_;
  SharedIndex* facs__simple_radial_fixed_focal__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_focal__args__extra_calib__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_focal__args__point__idx_shared_;
  float* facs__simple_radial_fixed_focal__args__pixel__data_;
  float* facs__simple_radial_fixed_focal__args__focal__data_;
  SharedIndex* facs__simple_radial_fixed_extra_calib__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_extra_calib__args__focal__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_extra_calib__args__point__idx_shared_;
  float* facs__simple_radial_fixed_extra_calib__args__pixel__data_;
  float* facs__simple_radial_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_point__args__focal__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_point__args__extra_calib__idx_shared_;
  float* facs__simple_radial_fixed_point__args__pixel__data_;
  float* facs__simple_radial_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal__args__point__idx_shared_;
  float* facs__simple_radial_fixed_pose_fixed_focal__args__pixel__data_;
  float* facs__simple_radial_fixed_pose_fixed_focal__args__pose__data_;
  float* facs__simple_radial_fixed_pose_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__idx_shared_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__args__pixel__data_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_point__args__focal__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__idx_shared_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__pose__data_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__idx_shared_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__args__pixel__data_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__args__focal__data_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__idx_shared_;
  float* facs__simple_radial_fixed_focal_fixed_point__args__pixel__data_;
  float* facs__simple_radial_fixed_focal_fixed_point__args__focal__data_;
  float* facs__simple_radial_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__idx_shared_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__focal__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole__args__pose__idx_shared_;
  SharedIndex* facs__pinhole__args__focal__idx_shared_;
  SharedIndex* facs__pinhole__args__extra_calib__idx_shared_;
  SharedIndex* facs__pinhole__args__point__idx_shared_;
  float* facs__pinhole__args__pixel__data_;
  SharedIndex* facs__pinhole_fixed_pose__args__focal__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose__args__extra_calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose__args__pixel__data_;
  float* facs__pinhole_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_fixed_focal__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_fixed_focal__args__extra_calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_focal__args__point__idx_shared_;
  float* facs__pinhole_fixed_focal__args__pixel__data_;
  float* facs__pinhole_fixed_focal__args__focal__data_;
  SharedIndex* facs__pinhole_fixed_extra_calib__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_fixed_extra_calib__args__focal__idx_shared_;
  SharedIndex* facs__pinhole_fixed_extra_calib__args__point__idx_shared_;
  float* facs__pinhole_fixed_extra_calib__args__pixel__data_;
  float* facs__pinhole_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex* facs__pinhole_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_fixed_point__args__focal__idx_shared_;
  SharedIndex* facs__pinhole_fixed_point__args__extra_calib__idx_shared_;
  float* facs__pinhole_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose_fixed_focal__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_focal__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_focal__args__pose__data_;
  float* facs__pinhole_fixed_pose_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_extra_calib__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__args__pose__data_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex* facs__pinhole_fixed_pose_fixed_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_point__args__extra_calib__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_point__args__pose__data_;
  float* facs__pinhole_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_focal_fixed_extra_calib__args__point__idx_shared_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__args__pixel__data_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__args__focal__data_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex* facs__pinhole_fixed_focal_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_focal_fixed_point__args__extra_calib__idx_shared_;
  float* facs__pinhole_fixed_focal_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_focal_fixed_point__args__focal__data_;
  float* facs__pinhole_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_extra_calib_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_extra_calib_fixed_point__args__focal__idx_shared_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__idx_shared_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pixel__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__focal__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__extra_calib__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__pose__data_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__focal__data_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__idx_shared_;
  float*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__idx_shared_;
  float*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__focal__data_;
  float*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__extra_calib__data_;
  float*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__point__data_;
  float* marker__scratch_inout_;
  float* facs__simple_radial__res_;
  float* facs__simple_radial_fixed_pose__res_;
  float* facs__simple_radial_fixed_focal__res_;
  float* facs__simple_radial_fixed_extra_calib__res_;
  float* facs__simple_radial_fixed_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_focal__res_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__res_;
  float* facs__simple_radial_fixed_pose_fixed_point__res_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__res_;
  float* facs__simple_radial_fixed_focal_fixed_point__res_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__res_;
  float* facs__simple_radial_fixed_pose_fixed_focal_fixed_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__res_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__res_;
  float* facs__pinhole__res_;
  float* facs__pinhole_fixed_pose__res_;
  float* facs__pinhole_fixed_focal__res_;
  float* facs__pinhole_fixed_extra_calib__res_;
  float* facs__pinhole_fixed_point__res_;
  float* facs__pinhole_fixed_pose_fixed_focal__res_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__res_;
  float* facs__pinhole_fixed_pose_fixed_point__res_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__res_;
  float* facs__pinhole_fixed_focal_fixed_point__res_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__res_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__res_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__res_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__res_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__res_;
  float* facs__simple_radial__args__pose__jac_;
  float* facs__simple_radial__args__focal__jac_;
  float* facs__simple_radial__args__extra_calib__jac_;
  float* facs__simple_radial__args__point__jac_;
  float* facs__simple_radial_fixed_pose__args__focal__jac_;
  float* facs__simple_radial_fixed_pose__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_pose__args__point__jac_;
  float* facs__simple_radial_fixed_focal__args__pose__jac_;
  float* facs__simple_radial_fixed_focal__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_focal__args__point__jac_;
  float* facs__simple_radial_fixed_extra_calib__args__pose__jac_;
  float* facs__simple_radial_fixed_extra_calib__args__focal__jac_;
  float* facs__simple_radial_fixed_extra_calib__args__point__jac_;
  float* facs__simple_radial_fixed_point__args__pose__jac_;
  float* facs__simple_radial_fixed_point__args__focal__jac_;
  float* facs__simple_radial_fixed_point__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_pose_fixed_focal__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_pose_fixed_focal__args__point__jac_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__args__focal__jac_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__args__point__jac_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__focal__jac_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__args__pose__jac_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__args__point__jac_;
  float* facs__simple_radial_fixed_focal_fixed_point__args__pose__jac_;
  float* facs__simple_radial_fixed_focal_fixed_point__args__extra_calib__jac_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__args__pose__jac_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__args__focal__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__args__point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_fixed_point__args__extra_calib__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__args__focal__jac_;
  float*
      facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__args__pose__jac_;
  float* facs__pinhole__args__pose__jac_;
  float* facs__pinhole__args__focal__jac_;
  float* facs__pinhole__args__extra_calib__jac_;
  float* facs__pinhole__args__point__jac_;
  float* facs__pinhole_fixed_pose__args__focal__jac_;
  float* facs__pinhole_fixed_pose__args__extra_calib__jac_;
  float* facs__pinhole_fixed_pose__args__point__jac_;
  float* facs__pinhole_fixed_focal__args__pose__jac_;
  float* facs__pinhole_fixed_focal__args__extra_calib__jac_;
  float* facs__pinhole_fixed_focal__args__point__jac_;
  float* facs__pinhole_fixed_extra_calib__args__pose__jac_;
  float* facs__pinhole_fixed_extra_calib__args__focal__jac_;
  float* facs__pinhole_fixed_extra_calib__args__point__jac_;
  float* facs__pinhole_fixed_point__args__pose__jac_;
  float* facs__pinhole_fixed_point__args__focal__jac_;
  float* facs__pinhole_fixed_point__args__extra_calib__jac_;
  float* facs__pinhole_fixed_pose_fixed_focal__args__extra_calib__jac_;
  float* facs__pinhole_fixed_pose_fixed_focal__args__point__jac_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__args__focal__jac_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__args__point__jac_;
  float* facs__pinhole_fixed_pose_fixed_point__args__focal__jac_;
  float* facs__pinhole_fixed_pose_fixed_point__args__extra_calib__jac_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__args__pose__jac_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__args__point__jac_;
  float* facs__pinhole_fixed_focal_fixed_point__args__pose__jac_;
  float* facs__pinhole_fixed_focal_fixed_point__args__extra_calib__jac_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__args__pose__jac_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__args__focal__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__args__point__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_fixed_point__args__extra_calib__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__args__focal__jac_;
  float*
      facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__args__pose__jac_;
  float* nodes__PinholeExtraCalib__z_;
  float* nodes__PinholeExtraCalib__z_end__;
  float* nodes__PinholeFocal__z_;
  float* nodes__PinholeFocal__z_end__;
  float* nodes__Point__z_;
  float* nodes__Point__z_end__;
  float* nodes__Pose__z_;
  float* nodes__Pose__z_end__;
  float* nodes__SimpleRadialExtraCalib__z_;
  float* nodes__SimpleRadialExtraCalib__z_end__;
  float* nodes__SimpleRadialFocal__z_;
  float* nodes__SimpleRadialFocal__z_end__;
  float* nodes__PinholeExtraCalib__p_;
  float* nodes__PinholeExtraCalib__p_end__;
  float* nodes__PinholeFocal__p_;
  float* nodes__PinholeFocal__p_end__;
  float* nodes__Point__p_;
  float* nodes__Point__p_end__;
  float* nodes__Pose__p_;
  float* nodes__Pose__p_end__;
  float* nodes__SimpleRadialExtraCalib__p_;
  float* nodes__SimpleRadialExtraCalib__p_end__;
  float* nodes__SimpleRadialFocal__p_;
  float* nodes__SimpleRadialFocal__p_end__;
  float* nodes__PinholeExtraCalib__step_;
  float* nodes__PinholeExtraCalib__step_end__;
  float* nodes__PinholeFocal__step_;
  float* nodes__PinholeFocal__step_end__;
  float* nodes__Point__step_;
  float* nodes__Point__step_end__;
  float* nodes__Pose__step_;
  float* nodes__Pose__step_end__;
  float* nodes__SimpleRadialExtraCalib__step_;
  float* nodes__SimpleRadialExtraCalib__step_end__;
  float* nodes__SimpleRadialFocal__step_;
  float* nodes__SimpleRadialFocal__step_end__;
  float* marker__w_start_;
  float* nodes__PinholeExtraCalib__w_;
  float* nodes__PinholeFocal__w_;
  float* nodes__Point__w_;
  float* nodes__Pose__w_;
  float* nodes__SimpleRadialExtraCalib__w_;
  float* nodes__SimpleRadialFocal__w_;
  float* marker__w_end_;
  float* marker__r_0_start_;
  float* nodes__PinholeExtraCalib__r_0_;
  float* nodes__PinholeFocal__r_0_;
  float* nodes__Point__r_0_;
  float* nodes__Pose__r_0_;
  float* nodes__SimpleRadialExtraCalib__r_0_;
  float* nodes__SimpleRadialFocal__r_0_;
  float* marker__r_0_end_;
  float* marker__r_k_start_;
  float* nodes__PinholeExtraCalib__r_k_;
  float* nodes__PinholeFocal__r_k_;
  float* nodes__Point__r_k_;
  float* nodes__Pose__r_k_;
  float* nodes__SimpleRadialExtraCalib__r_k_;
  float* nodes__SimpleRadialFocal__r_k_;
  float* marker__r_k_end_;
  float* marker__Mp_start_;
  float* nodes__PinholeExtraCalib__Mp_;
  float* nodes__PinholeFocal__Mp_;
  float* nodes__Point__Mp_;
  float* nodes__Pose__Mp_;
  float* nodes__SimpleRadialExtraCalib__Mp_;
  float* nodes__SimpleRadialFocal__Mp_;
  float* marker__Mp_end_;
  float* marker__precond_start_;
  float* nodes__PinholeExtraCalib__precond_diag_;
  float* nodes__PinholeExtraCalib__precond_tril_;
  float* nodes__PinholeFocal__precond_diag_;
  float* nodes__PinholeFocal__precond_tril_;
  float* nodes__Point__precond_diag_;
  float* nodes__Point__precond_tril_;
  float* nodes__Pose__precond_diag_;
  float* nodes__Pose__precond_tril_;
  float* nodes__SimpleRadialExtraCalib__precond_diag_;
  float* nodes__SimpleRadialExtraCalib__precond_tril_;
  float* nodes__SimpleRadialFocal__precond_diag_;
  float* nodes__SimpleRadialFocal__precond_tril_;
  float* marker__precond_end_;
  float* marker__jp_start_;
  float* facs__simple_radial__jp_;
  float* facs__simple_radial_fixed_pose__jp_;
  float* facs__simple_radial_fixed_focal__jp_;
  float* facs__simple_radial_fixed_extra_calib__jp_;
  float* facs__simple_radial_fixed_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_focal__jp_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib__jp_;
  float* facs__simple_radial_fixed_pose_fixed_point__jp_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib__jp_;
  float* facs__simple_radial_fixed_focal_fixed_point__jp_;
  float* facs__simple_radial_fixed_extra_calib_fixed_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_focal_fixed_extra_calib__jp_;
  float* facs__simple_radial_fixed_pose_fixed_focal_fixed_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_extra_calib_fixed_point__jp_;
  float* facs__simple_radial_fixed_focal_fixed_extra_calib_fixed_point__jp_;
  float* facs__pinhole__jp_;
  float* facs__pinhole_fixed_pose__jp_;
  float* facs__pinhole_fixed_focal__jp_;
  float* facs__pinhole_fixed_extra_calib__jp_;
  float* facs__pinhole_fixed_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_focal__jp_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib__jp_;
  float* facs__pinhole_fixed_pose_fixed_point__jp_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib__jp_;
  float* facs__pinhole_fixed_focal_fixed_point__jp_;
  float* facs__pinhole_fixed_extra_calib_fixed_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_extra_calib__jp_;
  float* facs__pinhole_fixed_pose_fixed_focal_fixed_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_extra_calib_fixed_point__jp_;
  float* facs__pinhole_fixed_focal_fixed_extra_calib_fixed_point__jp_;
  float* marker__jp_end_;
  float* solver__current_diag_;
  float* solver__alpha_numerator_;
  float* solver__alpha_denumerator_;
  float* solver__alpha_;
  float* solver__neg_alpha_;
  float* solver__beta_numerator_;
  float* solver__beta_;
  float* solver__r_0_norm2_tot_;
  float* solver__r_kp1_norm2_tot_;
  float* solver__pred_decrease_tot_;
  float* solver__res_tot_;
};

}  // namespace caspar