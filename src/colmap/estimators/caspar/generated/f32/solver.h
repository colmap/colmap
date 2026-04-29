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
   * @param PinholeCalib_num_max the maximum number of PinholeCalibs
   * @param PinholeFocalAndExtra_num_max the maximum number of
   * PinholeFocalAndExtras
   * @param PinholePose_num_max the maximum number of PinholePoses
   * @param PinholePrincipalPoint_num_max the maximum number of
   * PinholePrincipalPoints
   * @param Point_num_max the maximum number of Points
   * @param SimpleRadialCalib_num_max the maximum number of SimpleRadialCalibs
   * @param SimpleRadialFocalAndExtra_num_max the maximum number of
   * SimpleRadialFocalAndExtras
   * @param SimpleRadialPose_num_max the maximum number of SimpleRadialPoses
   * @param SimpleRadialPrincipalPoint_num_max the maximum number of
   * SimpleRadialPrincipalPoints
   * @param simple_radial_merged_num_max the maximum number of
   * simple_radial_mergeds
   * @param simple_radial_merged_fixed_pose_num_max the maximum number of
   * simple_radial_merged_fixed_poses
   * @param simple_radial_merged_fixed_point_num_max the maximum number of
   * simple_radial_merged_fixed_points
   * @param simple_radial_merged_fixed_pose_fixed_point_num_max the maximum
   * number of simple_radial_merged_fixed_pose_fixed_points
   * @param pinhole_merged_num_max the maximum number of pinhole_mergeds
   * @param pinhole_merged_fixed_pose_num_max the maximum number of
   * pinhole_merged_fixed_poses
   * @param pinhole_merged_fixed_point_num_max the maximum number of
   * pinhole_merged_fixed_points
   * @param pinhole_merged_fixed_pose_fixed_point_num_max the maximum number of
   * pinhole_merged_fixed_pose_fixed_points
   * @param simple_radial_fixed_focal_and_extra_num_max the maximum number of
   * simple_radial_fixed_focal_and_extras
   * @param simple_radial_fixed_principal_point_num_max the maximum number of
   * simple_radial_fixed_principal_points
   * @param simple_radial_fixed_pose_fixed_focal_and_extra_num_max the maximum
   * number of simple_radial_fixed_pose_fixed_focal_and_extras
   * @param simple_radial_fixed_pose_fixed_principal_point_num_max the maximum
   * number of simple_radial_fixed_pose_fixed_principal_points
   * @param simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max
   * the maximum number of
   * simple_radial_fixed_focal_and_extra_fixed_principal_points
   * @param simple_radial_fixed_focal_and_extra_fixed_point_num_max the maximum
   * number of simple_radial_fixed_focal_and_extra_fixed_points
   * @param simple_radial_fixed_principal_point_fixed_point_num_max the maximum
   * number of simple_radial_fixed_principal_point_fixed_points
   * @param
   * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max
   * the maximum number of
   * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_points
   * @param simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max
   * the maximum number of
   * simple_radial_fixed_pose_fixed_focal_and_extra_fixed_points
   * @param simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_fixed_pose_fixed_principal_point_fixed_points
   * @param
   * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_points
   * @param pinhole_fixed_focal_and_extra_num_max the maximum number of
   * pinhole_fixed_focal_and_extras
   * @param pinhole_fixed_principal_point_num_max the maximum number of
   * pinhole_fixed_principal_points
   * @param pinhole_fixed_pose_fixed_focal_and_extra_num_max the maximum number
   * of pinhole_fixed_pose_fixed_focal_and_extras
   * @param pinhole_fixed_pose_fixed_principal_point_num_max the maximum number
   * of pinhole_fixed_pose_fixed_principal_points
   * @param pinhole_fixed_focal_and_extra_fixed_principal_point_num_max the
   * maximum number of pinhole_fixed_focal_and_extra_fixed_principal_points
   * @param pinhole_fixed_focal_and_extra_fixed_point_num_max the maximum number
   * of pinhole_fixed_focal_and_extra_fixed_points
   * @param pinhole_fixed_principal_point_fixed_point_num_max the maximum number
   * of pinhole_fixed_principal_point_fixed_points
   * @param
   * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max the
   * maximum number of
   * pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_points
   * @param pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max the
   * maximum number of pinhole_fixed_pose_fixed_focal_and_extra_fixed_points
   * @param pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max the
   * maximum number of pinhole_fixed_pose_fixed_principal_point_fixed_points
   * @param
   * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max the
   * maximum number of
   * pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_points
   */
  GraphSolver(
      const SolverParams<double>& params,
      size_t PinholeCalib_num_max,
      size_t PinholeFocalAndExtra_num_max,
      size_t PinholePose_num_max,
      size_t PinholePrincipalPoint_num_max,
      size_t Point_num_max,
      size_t SimpleRadialCalib_num_max,
      size_t SimpleRadialFocalAndExtra_num_max,
      size_t SimpleRadialPose_num_max,
      size_t SimpleRadialPrincipalPoint_num_max,
      size_t simple_radial_merged_num_max,
      size_t simple_radial_merged_fixed_pose_num_max,
      size_t simple_radial_merged_fixed_point_num_max,
      size_t simple_radial_merged_fixed_pose_fixed_point_num_max,
      size_t pinhole_merged_num_max,
      size_t pinhole_merged_fixed_pose_num_max,
      size_t pinhole_merged_fixed_point_num_max,
      size_t pinhole_merged_fixed_pose_fixed_point_num_max,
      size_t simple_radial_fixed_focal_and_extra_num_max,
      size_t simple_radial_fixed_principal_point_num_max,
      size_t simple_radial_fixed_pose_fixed_focal_and_extra_num_max,
      size_t simple_radial_fixed_pose_fixed_principal_point_num_max,
      size_t simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t simple_radial_fixed_focal_and_extra_fixed_point_num_max,
      size_t simple_radial_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_fixed_focal_and_extra_num_max,
      size_t pinhole_fixed_principal_point_num_max,
      size_t pinhole_fixed_pose_fixed_focal_and_extra_num_max,
      size_t pinhole_fixed_pose_fixed_principal_point_num_max,
      size_t pinhole_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t pinhole_fixed_focal_and_extra_fixed_point_num_max,
      size_t pinhole_fixed_principal_point_fixed_point_num_max,
      size_t
          pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max);

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
   * Set the current value for the PinholeCalib nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeCalibNodesFromStackedHost(const float* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeCalib nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeCalibNodesFromStackedDevice(const float* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedHost(float* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedDevice(float* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current number of active nodes of type PinholeCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeCalibNum(size_t num);

  /**
   * Set the current value for the PinholeFocalAndExtra nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalAndExtraNodesFromStackedHost(const float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the current value for the PinholeFocalAndExtra nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalAndExtraNodesFromStackedDevice(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Read the current value for the PinholeFocalAndExtra nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalAndExtraNodesToStackedHost(float* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Read the current value for the PinholeFocalAndExtra nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalAndExtraNodesToStackedDevice(float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the current number of active nodes of type PinholeFocalAndExtra.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFocalAndExtraNum(size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedHost(const float* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedDevice(const float* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedHost(float* const data,
                                        size_t offset,
                                        size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedDevice(float* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the current number of active nodes of type PinholePose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholePoseNum(size_t num);

  /**
   * Set the current value for the PinholePrincipalPoint nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePrincipalPointNodesFromStackedHost(const float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current value for the PinholePrincipalPoint nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePrincipalPointNodesFromStackedDevice(const float* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedHost(float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedDevice(float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current number of active nodes of type PinholePrincipalPoint.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholePrincipalPointNum(size_t num);

  /**
   * Set the current value for the Point nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedHost(const float* const data,
                                    size_t offset,
                                    size_t num);

  /**
   * Set the current value for the Point nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedDevice(const float* const data,
                                      size_t offset,
                                      size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedHost(float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedDevice(float* const data,
                                    size_t offset,
                                    size_t num);

  /**
   * Set the current number of active nodes of type Point.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPointNum(size_t num);

  /**
   * Set the current value for the SimpleRadialCalib nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialCalibNodesFromStackedHost(const float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current value for the SimpleRadialCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialCalibNodesFromStackedDevice(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedHost(float* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedDevice(float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialCalibNum(size_t num);

  /**
   * Set the current value for the SimpleRadialFocalAndExtra nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialFocalAndExtraNodesFromStackedHost(const float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Set the current value for the SimpleRadialFocalAndExtra nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialFocalAndExtraNodesFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndExtra nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndExtraNodesToStackedHost(float* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndExtra nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndExtraNodesToStackedDevice(float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialFocalAndExtra.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFocalAndExtraNum(size_t num);

  /**
   * Set the current value for the SimpleRadialPose nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPoseNodesFromStackedHost(const float* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the current value for the SimpleRadialPose nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPoseNodesFromStackedDevice(const float* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedHost(float* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedDevice(float* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialPose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialPoseNum(size_t num);

  /**
   * Set the current value for the SimpleRadialPrincipalPoint nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPrincipalPointNodesFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialPrincipalPoint nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPrincipalPointNodesFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedHost(float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedDevice(float* const data,
                                                         size_t offset,
                                                         size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialPrincipalPoint.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialMerged factor
   * from host.
   */
  void SetSimpleRadialMergedPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialMerged factor
   * from device.
   */
  void SetSimpleRadialMergedPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMerged factor
   * from host.
   */
  void SetSimpleRadialMergedCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMerged factor
   * from device.
   */
  void SetSimpleRadialMergedCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialMerged factor
   * from host.
   */
  void SetSimpleRadialMergedPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialMerged factor
   * from device.
   */
  void SetSimpleRadialMergedPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMerged factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedPixelDataFromStackedHost(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMerged factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedPixelDataFromStackedDevice(const float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the current number of SimpleRadialMerged factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialMergedNum(size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMergedFixedPose
   * factor from host.
   */
  void SetSimpleRadialMergedFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMergedFixedPose
   * factor from device.
   */
  void SetSimpleRadialMergedFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialMergedFixedPose
   * factor from host.
   */
  void SetSimpleRadialMergedFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialMergedFixedPose
   * factor from device.
   */
  void SetSimpleRadialMergedFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPosePixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPosePixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialMergedFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPosePoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialMergedFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPosePoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialMergedFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialMergedFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialMergedFixedPoint
   * factor from host.
   */
  void SetSimpleRadialMergedFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialMergedFixedPoint
   * factor from device.
   */
  void SetSimpleRadialMergedFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMergedFixedPoint
   * factor from host.
   */
  void SetSimpleRadialMergedFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialMergedFixedPoint
   * factor from device.
   */
  void SetSimpleRadialMergedFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialMergedFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialMergedFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialMergedFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialMergedFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialMergedFixedPoseFixedPoint factor from host.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialMergedFixedPoseFixedPoint factor from device.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialMergedFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialMergedFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialMergedFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeMerged factor from
   * host.
   */
  void SetPinholeMergedPoseIndicesFromHost(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the pose argument for the PinholeMerged factor from
   * device.
   */
  void SetPinholeMergedPoseIndicesFromDevice(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMerged factor from
   * host.
   */
  void SetPinholeMergedCalibIndicesFromHost(const unsigned int* const indices,
                                            size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMerged factor from
   * device.
   */
  void SetPinholeMergedCalibIndicesFromDevice(const unsigned int* const indices,
                                              size_t num);

  /**
   * Set the indices for the point argument for the PinholeMerged factor from
   * host.
   */
  void SetPinholeMergedPointIndicesFromHost(const unsigned int* const indices,
                                            size_t num);

  /**
   * Set the indices for the point argument for the PinholeMerged factor from
   * device.
   */
  void SetPinholeMergedPointIndicesFromDevice(const unsigned int* const indices,
                                              size_t num);

  /**
   * Set the values for the pixel consts PinholeMerged factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedPixelDataFromStackedHost(const float* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the values for the pixel consts PinholeMerged factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedPixelDataFromStackedDevice(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the current number of PinholeMerged factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeMergedNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMergedFixedPose
   * factor from host.
   */
  void SetPinholeMergedFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMergedFixedPose
   * factor from device.
   */
  void SetPinholeMergedFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeMergedFixedPose
   * factor from host.
   */
  void SetPinholeMergedFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeMergedFixedPose
   * factor from device.
   */
  void SetPinholeMergedFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPosePixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPosePixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeMergedFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPosePoseDataFromStackedHost(const float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Set the values for the pose consts PinholeMergedFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPosePoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeMergedFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeMergedFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeMergedFixedPoint
   * factor from host.
   */
  void SetPinholeMergedFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeMergedFixedPoint
   * factor from device.
   */
  void SetPinholeMergedFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMergedFixedPoint
   * factor from host.
   */
  void SetPinholeMergedFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeMergedFixedPoint
   * factor from device.
   */
  void SetPinholeMergedFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeMergedFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeMergedFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeMergedFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeMergedFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the
   * PinholeMergedFixedPoseFixedPoint factor from host.
   */
  void SetPinholeMergedFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * PinholeMergedFixedPoseFixedPoint factor from device.
   */
  void SetPinholeMergedFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeMergedFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeMergedFixedPoseFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeMergedFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeMergedFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialFixedFocalAndExtraPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedFocalAndExtra factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedFocalAndExtra factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from device.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoseFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoseFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedFocalAndExtraFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from device.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedFocalAndExtraFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedFocalAndExtraFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedPrincipalPointFixedPoint factor from host.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedPrincipalPointFixedPoint factor from device.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPrincipalPointFixedPoint factor from host.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * host.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * device.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from host.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoseFixedFocalAndExtraFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * host.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * device.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedFocalAndExtra
   * factor from host.
   */
  void SetPinholeFixedFocalAndExtraPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedFocalAndExtra
   * factor from device.
   */
  void SetPinholeFixedFocalAndExtraPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedFocalAndExtra factor from host.
   */
  void SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedFocalAndExtra factor from device.
   */
  void SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedFocalAndExtra
   * factor from host.
   */
  void SetPinholeFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedFocalAndExtra
   * factor from device.
   */
  void SetPinholeFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedFocalAndExtra factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedFocalAndExtra factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts PinholeFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts PinholeFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPrincipalPoint
   * factor from host.
   */
  void SetPinholeFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPrincipalPoint
   * factor from device.
   */
  void SetPinholeFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPrincipalPoint factor from host.
   */
  void SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPrincipalPoint factor from device.
   */
  void SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPrincipalPoint
   * factor from host.
   */
  void SetPinholeFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPrincipalPoint
   * factor from device.
   */
  void SetPinholeFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts PinholeFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts PinholeFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedPoseFixedFocalAndExtra factor from device.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedFocalAndExtra factor from device.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedFocalAndExtraFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPoint factor from device.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedFocalAndExtraFixedPoint factor from device.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedFocalAndExtraFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedFocalAndExtraFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedFocalAndExtraFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedFocalAndExtraFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedFocalAndExtraFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeFixedPoseFixedFocalAndExtraFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from host.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedPoseFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedFocalAndExtraFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedPoseFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeFixedFocalAndExtraFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointNum(size_t num);

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

  size_t PinholeCalib_num_;
  size_t PinholeCalib_num_max_;
  size_t PinholeFocalAndExtra_num_;
  size_t PinholeFocalAndExtra_num_max_;
  size_t PinholePose_num_;
  size_t PinholePose_num_max_;
  size_t PinholePrincipalPoint_num_;
  size_t PinholePrincipalPoint_num_max_;
  size_t Point_num_;
  size_t Point_num_max_;
  size_t SimpleRadialCalib_num_;
  size_t SimpleRadialCalib_num_max_;
  size_t SimpleRadialFocalAndExtra_num_;
  size_t SimpleRadialFocalAndExtra_num_max_;
  size_t SimpleRadialPose_num_;
  size_t SimpleRadialPose_num_max_;
  size_t SimpleRadialPrincipalPoint_num_;
  size_t SimpleRadialPrincipalPoint_num_max_;
  size_t simple_radial_merged_num_;
  size_t simple_radial_merged_num_max_;
  size_t simple_radial_merged_fixed_pose_num_;
  size_t simple_radial_merged_fixed_pose_num_max_;
  size_t simple_radial_merged_fixed_point_num_;
  size_t simple_radial_merged_fixed_point_num_max_;
  size_t simple_radial_merged_fixed_pose_fixed_point_num_;
  size_t simple_radial_merged_fixed_pose_fixed_point_num_max_;
  size_t pinhole_merged_num_;
  size_t pinhole_merged_num_max_;
  size_t pinhole_merged_fixed_pose_num_;
  size_t pinhole_merged_fixed_pose_num_max_;
  size_t pinhole_merged_fixed_point_num_;
  size_t pinhole_merged_fixed_point_num_max_;
  size_t pinhole_merged_fixed_pose_fixed_point_num_;
  size_t pinhole_merged_fixed_pose_fixed_point_num_max_;
  size_t simple_radial_fixed_focal_and_extra_num_;
  size_t simple_radial_fixed_focal_and_extra_num_max_;
  size_t simple_radial_fixed_principal_point_num_;
  size_t simple_radial_fixed_principal_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_focal_and_extra_num_;
  size_t simple_radial_fixed_pose_fixed_focal_and_extra_num_max_;
  size_t simple_radial_fixed_pose_fixed_principal_point_num_;
  size_t simple_radial_fixed_pose_fixed_principal_point_num_max_;
  size_t simple_radial_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t simple_radial_fixed_focal_and_extra_fixed_point_num_;
  size_t simple_radial_fixed_focal_and_extra_fixed_point_num_max_;
  size_t simple_radial_fixed_principal_point_fixed_point_num_;
  size_t simple_radial_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_;
  size_t
      simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_fixed_focal_and_extra_num_;
  size_t pinhole_fixed_focal_and_extra_num_max_;
  size_t pinhole_fixed_principal_point_num_;
  size_t pinhole_fixed_principal_point_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_and_extra_num_;
  size_t pinhole_fixed_pose_fixed_focal_and_extra_num_max_;
  size_t pinhole_fixed_pose_fixed_principal_point_num_;
  size_t pinhole_fixed_pose_fixed_principal_point_num_max_;
  size_t pinhole_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t pinhole_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t pinhole_fixed_focal_and_extra_fixed_point_num_;
  size_t pinhole_fixed_focal_and_extra_fixed_point_num_max_;
  size_t pinhole_fixed_principal_point_fixed_point_num_;
  size_t pinhole_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_;
  size_t
      pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_;

  size_t get_nbytes();
  float LinearizeFirst();
  void Linearize();
  float DoResJacFirst();
  void DoResJac();
  void DoNormalize();
  void DoJtjpDirect();
  void DoAlphaFirst();
  void DoAlpha();
  void DoUpdateStepFirst();
  void DoUpdateStep();
  void DoUpdateRFirst();
  void DoUpdateR();
  float DoRetractScore();
  void DoBeta();
  void DoUpdateP();
  void DoUpdateMp();
  float GetPredDecrease();

  float* marker__start_;
  float* nodes__PinholeCalib__storage_current_;
  float* nodes__PinholeCalib__storage_check_;
  float* nodes__PinholeCalib__storage_new_best_;
  float* nodes__PinholeFocalAndExtra__storage_current_;
  float* nodes__PinholeFocalAndExtra__storage_check_;
  float* nodes__PinholeFocalAndExtra__storage_new_best_;
  float* nodes__PinholePose__storage_current_;
  float* nodes__PinholePose__storage_check_;
  float* nodes__PinholePose__storage_new_best_;
  float* nodes__PinholePrincipalPoint__storage_current_;
  float* nodes__PinholePrincipalPoint__storage_check_;
  float* nodes__PinholePrincipalPoint__storage_new_best_;
  float* nodes__Point__storage_current_;
  float* nodes__Point__storage_check_;
  float* nodes__Point__storage_new_best_;
  float* nodes__SimpleRadialCalib__storage_current_;
  float* nodes__SimpleRadialCalib__storage_check_;
  float* nodes__SimpleRadialCalib__storage_new_best_;
  float* nodes__SimpleRadialFocalAndExtra__storage_current_;
  float* nodes__SimpleRadialFocalAndExtra__storage_check_;
  float* nodes__SimpleRadialFocalAndExtra__storage_new_best_;
  float* nodes__SimpleRadialPose__storage_current_;
  float* nodes__SimpleRadialPose__storage_check_;
  float* nodes__SimpleRadialPose__storage_new_best_;
  float* nodes__SimpleRadialPrincipalPoint__storage_current_;
  float* nodes__SimpleRadialPrincipalPoint__storage_check_;
  float* nodes__SimpleRadialPrincipalPoint__storage_new_best_;
  SharedIndex* facs__simple_radial_merged__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_merged__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial_merged__args__point__idx_shared_;
  float* facs__simple_radial_merged__args__pixel__data_;
  SharedIndex* facs__simple_radial_merged_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial_merged_fixed_pose__args__point__idx_shared_;
  float* facs__simple_radial_merged_fixed_pose__args__pixel__data_;
  float* facs__simple_radial_merged_fixed_pose__args__pose__data_;
  SharedIndex* facs__simple_radial_merged_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_merged_fixed_point__args__calib__idx_shared_;
  float* facs__simple_radial_merged_fixed_point__args__pixel__data_;
  float* facs__simple_radial_merged_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__idx_shared_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__args__pose__data_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_merged__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_merged__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_merged__args__point__idx_shared_;
  float* facs__pinhole_merged__args__pixel__data_;
  SharedIndex* facs__pinhole_merged_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_merged_fixed_pose__args__point__idx_shared_;
  float* facs__pinhole_merged_fixed_pose__args__pixel__data_;
  float* facs__pinhole_merged_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_merged_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_merged_fixed_point__args__calib__idx_shared_;
  float* facs__pinhole_merged_fixed_point__args__pixel__data_;
  float* facs__pinhole_merged_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_merged_fixed_pose_fixed_point__args__calib__idx_shared_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__args__pose__data_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra__args__point__idx_shared_;
  float* facs__simple_radial_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__simple_radial_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_principal_point__args__point__idx_shared_;
  float* facs__simple_radial_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__simple_radial_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_fixed_focal_and_extra__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex* facs__pinhole_fixed_focal_and_extra__args__point__idx_shared_;
  float* facs__pinhole_fixed_focal_and_extra__args__pixel__data_;
  float* facs__pinhole_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex* facs__pinhole_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex* facs__pinhole_fixed_principal_point__args__point__idx_shared_;
  float* facs__pinhole_fixed_principal_point__args__pixel__data_;
  float* facs__pinhole_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_principal_point__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_principal_point__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float* facs__pinhole_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_principal_point_fixed_point__args__principal_point__data_;
  float* facs__pinhole_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_;
  float* marker__scratch_inout_;
  float* facs__simple_radial_merged__res_;
  float* facs__simple_radial_merged_fixed_pose__res_;
  float* facs__simple_radial_merged_fixed_point__res_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__res_;
  float* facs__pinhole_merged__res_;
  float* facs__pinhole_merged_fixed_pose__res_;
  float* facs__pinhole_merged_fixed_point__res_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__res_;
  float* facs__simple_radial_fixed_focal_and_extra__res_;
  float* facs__simple_radial_fixed_principal_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_focal_and_extra__res_;
  float* facs__simple_radial_fixed_pose_fixed_principal_point__res_;
  float* facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__simple_radial_fixed_focal_and_extra_fixed_point__res_;
  float* facs__simple_radial_fixed_principal_point_fixed_point__res_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__res_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_;
  float* facs__pinhole_fixed_focal_and_extra__res_;
  float* facs__pinhole_fixed_principal_point__res_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra__res_;
  float* facs__pinhole_fixed_pose_fixed_principal_point__res_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_point__res_;
  float* facs__pinhole_fixed_principal_point_fixed_point__res_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__res_;
  float* facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__res_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_;
  float* facs__simple_radial_merged__args__pose__jac_;
  float* facs__simple_radial_merged__args__calib__jac_;
  float* facs__simple_radial_merged__args__point__jac_;
  float* facs__simple_radial_merged_fixed_pose__args__calib__jac_;
  float* facs__simple_radial_merged_fixed_pose__args__point__jac_;
  float* facs__simple_radial_merged_fixed_point__args__pose__jac_;
  float* facs__simple_radial_merged_fixed_point__args__calib__jac_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__args__calib__jac_;
  float* facs__pinhole_merged__args__pose__jac_;
  float* facs__pinhole_merged__args__calib__jac_;
  float* facs__pinhole_merged__args__point__jac_;
  float* facs__pinhole_merged_fixed_pose__args__calib__jac_;
  float* facs__pinhole_merged_fixed_pose__args__point__jac_;
  float* facs__pinhole_merged_fixed_point__args__pose__jac_;
  float* facs__pinhole_merged_fixed_point__args__calib__jac_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__args__calib__jac_;
  float* facs__simple_radial_fixed_focal_and_extra__args__pose__jac_;
  float* facs__simple_radial_fixed_focal_and_extra__args__principal_point__jac_;
  float* facs__simple_radial_fixed_focal_and_extra__args__point__jac_;
  float* facs__simple_radial_fixed_principal_point__args__pose__jac_;
  float* facs__simple_radial_fixed_principal_point__args__focal_and_extra__jac_;
  float* facs__simple_radial_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra__args__point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__pose__jac_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__simple_radial_fixed_principal_point_fixed_point__args__pose__jac_;
  float*
      facs__simple_radial_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_;
  float* facs__pinhole_fixed_focal_and_extra__args__pose__jac_;
  float* facs__pinhole_fixed_focal_and_extra__args__principal_point__jac_;
  float* facs__pinhole_fixed_focal_and_extra__args__point__jac_;
  float* facs__pinhole_fixed_principal_point__args__pose__jac_;
  float* facs__pinhole_fixed_principal_point__args__focal_and_extra__jac_;
  float* facs__pinhole_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra__args__point__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_;
  float* facs__pinhole_fixed_pose_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_point__args__pose__jac_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float* facs__pinhole_fixed_principal_point_fixed_point__args__pose__jac_;
  float*
      facs__pinhole_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_;
  float* nodes__PinholeCalib__z_;
  float* nodes__PinholeCalib__z_end__;
  float* nodes__PinholeFocalAndExtra__z_;
  float* nodes__PinholeFocalAndExtra__z_end__;
  float* nodes__PinholePose__z_;
  float* nodes__PinholePose__z_end__;
  float* nodes__PinholePrincipalPoint__z_;
  float* nodes__PinholePrincipalPoint__z_end__;
  float* nodes__Point__z_;
  float* nodes__Point__z_end__;
  float* nodes__SimpleRadialCalib__z_;
  float* nodes__SimpleRadialCalib__z_end__;
  float* nodes__SimpleRadialFocalAndExtra__z_;
  float* nodes__SimpleRadialFocalAndExtra__z_end__;
  float* nodes__SimpleRadialPose__z_;
  float* nodes__SimpleRadialPose__z_end__;
  float* nodes__SimpleRadialPrincipalPoint__z_;
  float* nodes__SimpleRadialPrincipalPoint__z_end__;
  float* nodes__PinholeCalib__p_;
  float* nodes__PinholeCalib__p_end__;
  float* nodes__PinholeFocalAndExtra__p_;
  float* nodes__PinholeFocalAndExtra__p_end__;
  float* nodes__PinholePose__p_;
  float* nodes__PinholePose__p_end__;
  float* nodes__PinholePrincipalPoint__p_;
  float* nodes__PinholePrincipalPoint__p_end__;
  float* nodes__Point__p_;
  float* nodes__Point__p_end__;
  float* nodes__SimpleRadialCalib__p_;
  float* nodes__SimpleRadialCalib__p_end__;
  float* nodes__SimpleRadialFocalAndExtra__p_;
  float* nodes__SimpleRadialFocalAndExtra__p_end__;
  float* nodes__SimpleRadialPose__p_;
  float* nodes__SimpleRadialPose__p_end__;
  float* nodes__SimpleRadialPrincipalPoint__p_;
  float* nodes__SimpleRadialPrincipalPoint__p_end__;
  float* nodes__PinholeCalib__step_;
  float* nodes__PinholeCalib__step_end__;
  float* nodes__PinholeFocalAndExtra__step_;
  float* nodes__PinholeFocalAndExtra__step_end__;
  float* nodes__PinholePose__step_;
  float* nodes__PinholePose__step_end__;
  float* nodes__PinholePrincipalPoint__step_;
  float* nodes__PinholePrincipalPoint__step_end__;
  float* nodes__Point__step_;
  float* nodes__Point__step_end__;
  float* nodes__SimpleRadialCalib__step_;
  float* nodes__SimpleRadialCalib__step_end__;
  float* nodes__SimpleRadialFocalAndExtra__step_;
  float* nodes__SimpleRadialFocalAndExtra__step_end__;
  float* nodes__SimpleRadialPose__step_;
  float* nodes__SimpleRadialPose__step_end__;
  float* nodes__SimpleRadialPrincipalPoint__step_;
  float* nodes__SimpleRadialPrincipalPoint__step_end__;
  float* marker__w_start_;
  float* nodes__PinholeCalib__w_;
  float* nodes__PinholeFocalAndExtra__w_;
  float* nodes__PinholePose__w_;
  float* nodes__PinholePrincipalPoint__w_;
  float* nodes__Point__w_;
  float* nodes__SimpleRadialCalib__w_;
  float* nodes__SimpleRadialFocalAndExtra__w_;
  float* nodes__SimpleRadialPose__w_;
  float* nodes__SimpleRadialPrincipalPoint__w_;
  float* marker__w_end_;
  float* marker__r_0_start_;
  float* nodes__PinholeCalib__r_0_;
  float* nodes__PinholeFocalAndExtra__r_0_;
  float* nodes__PinholePose__r_0_;
  float* nodes__PinholePrincipalPoint__r_0_;
  float* nodes__Point__r_0_;
  float* nodes__SimpleRadialCalib__r_0_;
  float* nodes__SimpleRadialFocalAndExtra__r_0_;
  float* nodes__SimpleRadialPose__r_0_;
  float* nodes__SimpleRadialPrincipalPoint__r_0_;
  float* marker__r_0_end_;
  float* marker__r_k_start_;
  float* nodes__PinholeCalib__r_k_;
  float* nodes__PinholeFocalAndExtra__r_k_;
  float* nodes__PinholePose__r_k_;
  float* nodes__PinholePrincipalPoint__r_k_;
  float* nodes__Point__r_k_;
  float* nodes__SimpleRadialCalib__r_k_;
  float* nodes__SimpleRadialFocalAndExtra__r_k_;
  float* nodes__SimpleRadialPose__r_k_;
  float* nodes__SimpleRadialPrincipalPoint__r_k_;
  float* marker__r_k_end_;
  float* marker__Mp_start_;
  float* nodes__PinholeCalib__Mp_;
  float* nodes__PinholeFocalAndExtra__Mp_;
  float* nodes__PinholePose__Mp_;
  float* nodes__PinholePrincipalPoint__Mp_;
  float* nodes__Point__Mp_;
  float* nodes__SimpleRadialCalib__Mp_;
  float* nodes__SimpleRadialFocalAndExtra__Mp_;
  float* nodes__SimpleRadialPose__Mp_;
  float* nodes__SimpleRadialPrincipalPoint__Mp_;
  float* marker__Mp_end_;
  float* marker__precond_start_;
  float* nodes__PinholeCalib__precond_diag_;
  float* nodes__PinholeCalib__precond_tril_;
  float* nodes__PinholeFocalAndExtra__precond_diag_;
  float* nodes__PinholeFocalAndExtra__precond_tril_;
  float* nodes__PinholePose__precond_diag_;
  float* nodes__PinholePose__precond_tril_;
  float* nodes__PinholePrincipalPoint__precond_diag_;
  float* nodes__PinholePrincipalPoint__precond_tril_;
  float* nodes__Point__precond_diag_;
  float* nodes__Point__precond_tril_;
  float* nodes__SimpleRadialCalib__precond_diag_;
  float* nodes__SimpleRadialCalib__precond_tril_;
  float* nodes__SimpleRadialFocalAndExtra__precond_diag_;
  float* nodes__SimpleRadialFocalAndExtra__precond_tril_;
  float* nodes__SimpleRadialPose__precond_diag_;
  float* nodes__SimpleRadialPose__precond_tril_;
  float* nodes__SimpleRadialPrincipalPoint__precond_diag_;
  float* nodes__SimpleRadialPrincipalPoint__precond_tril_;
  float* marker__precond_end_;
  float* marker__jp_start_;
  float* facs__simple_radial_merged__jp_;
  float* facs__simple_radial_merged_fixed_pose__jp_;
  float* facs__simple_radial_merged_fixed_point__jp_;
  float* facs__simple_radial_merged_fixed_pose_fixed_point__jp_;
  float* facs__pinhole_merged__jp_;
  float* facs__pinhole_merged_fixed_pose__jp_;
  float* facs__pinhole_merged_fixed_point__jp_;
  float* facs__pinhole_merged_fixed_pose_fixed_point__jp_;
  float* facs__simple_radial_fixed_focal_and_extra__jp_;
  float* facs__simple_radial_fixed_principal_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_focal_and_extra__jp_;
  float* facs__simple_radial_fixed_pose_fixed_principal_point__jp_;
  float* facs__simple_radial_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__simple_radial_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__simple_radial_fixed_principal_point_fixed_point__jp_;
  float*
      facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_principal_point_fixed_point__jp_;
  float*
      facs__simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_;
  float* facs__pinhole_fixed_focal_and_extra__jp_;
  float* facs__pinhole_fixed_principal_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra__jp_;
  float* facs__pinhole_fixed_pose_fixed_principal_point__jp_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__pinhole_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__pinhole_fixed_principal_point_fixed_point__jp_;
  float*
      facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_principal_point_fixed_point__jp_;
  float*
      facs__pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_;
  float* marker__jp_end_;
  float* solver__current_diag_;
  float* solver__alpha_numerator_;
  float* solver__alpha_denominator_;
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