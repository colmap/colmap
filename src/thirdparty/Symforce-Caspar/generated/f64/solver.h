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
   * @param PinholeFocal_num_max the maximum number of PinholeFocals
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
   * @param simple_radial_num_max the maximum number of simple_radials
   * @param simple_radial_fixed_pose_num_max the maximum number of
   * simple_radial_fixed_poses
   * @param simple_radial_fixed_point_num_max the maximum number of
   * simple_radial_fixed_points
   * @param simple_radial_fixed_pose_fixed_point_num_max the maximum number of
   * simple_radial_fixed_pose_fixed_points
   * @param pinhole_num_max the maximum number of pinholes
   * @param pinhole_fixed_pose_num_max the maximum number of pinhole_fixed_poses
   * @param pinhole_fixed_point_num_max the maximum number of
   * pinhole_fixed_points
   * @param pinhole_fixed_pose_fixed_point_num_max the maximum number of
   * pinhole_fixed_pose_fixed_points
   * @param simple_radial_split_fixed_focal_and_extra_num_max the maximum number
   * of simple_radial_split_fixed_focal_and_extras
   * @param simple_radial_split_fixed_principal_point_num_max the maximum number
   * of simple_radial_split_fixed_principal_points
   * @param simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max the
   * maximum number of simple_radial_split_fixed_pose_fixed_focal_and_extras
   * @param simple_radial_split_fixed_pose_fixed_principal_point_num_max the
   * maximum number of simple_radial_split_fixed_pose_fixed_principal_points
   * @param
   * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max the
   * maximum number of
   * simple_radial_split_fixed_focal_and_extra_fixed_principal_points
   * @param simple_radial_split_fixed_focal_and_extra_fixed_point_num_max the
   * maximum number of simple_radial_split_fixed_focal_and_extra_fixed_points
   * @param simple_radial_split_fixed_principal_point_fixed_point_num_max the
   * maximum number of simple_radial_split_fixed_principal_point_fixed_points
   * @param
   * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_points
   * @param
   * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_points
   * @param
   * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_principal_point_fixed_points
   * @param
   * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_focal_num_max the maximum number of
   * pinhole_split_fixed_focals
   * @param pinhole_split_fixed_principal_point_num_max the maximum number of
   * pinhole_split_fixed_principal_points
   * @param pinhole_split_fixed_pose_fixed_focal_num_max the maximum number of
   * pinhole_split_fixed_pose_fixed_focals
   * @param pinhole_split_fixed_pose_fixed_principal_point_num_max the maximum
   * number of pinhole_split_fixed_pose_fixed_principal_points
   * @param pinhole_split_fixed_focal_fixed_principal_point_num_max the maximum
   * number of pinhole_split_fixed_focal_fixed_principal_points
   * @param pinhole_split_fixed_focal_fixed_point_num_max the maximum number of
   * pinhole_split_fixed_focal_fixed_points
   * @param pinhole_split_fixed_principal_point_fixed_point_num_max the maximum
   * number of pinhole_split_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max
   * the maximum number of
   * pinhole_split_fixed_pose_fixed_focal_fixed_principal_points
   * @param pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max the maximum
   * number of pinhole_split_fixed_pose_fixed_focal_fixed_points
   * @param pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * pinhole_split_fixed_pose_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * pinhole_split_fixed_focal_fixed_principal_point_fixed_points
   */
  GraphSolver(
      const SolverParams<double>& params,
      size_t PinholeCalib_num_max,
      size_t PinholeFocal_num_max,
      size_t PinholePose_num_max,
      size_t PinholePrincipalPoint_num_max,
      size_t Point_num_max,
      size_t SimpleRadialCalib_num_max,
      size_t SimpleRadialFocalAndExtra_num_max,
      size_t SimpleRadialPose_num_max,
      size_t SimpleRadialPrincipalPoint_num_max,
      size_t simple_radial_num_max,
      size_t simple_radial_fixed_pose_num_max,
      size_t simple_radial_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_point_num_max,
      size_t pinhole_num_max,
      size_t pinhole_fixed_pose_num_max,
      size_t pinhole_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_point_num_max,
      size_t simple_radial_split_fixed_focal_and_extra_num_max,
      size_t simple_radial_split_fixed_principal_point_num_max,
      size_t simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max,
      size_t simple_radial_split_fixed_pose_fixed_principal_point_num_max,
      size_t
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t simple_radial_split_fixed_focal_and_extra_fixed_point_num_max,
      size_t simple_radial_split_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_fixed_focal_num_max,
      size_t pinhole_split_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_num_max,
      size_t pinhole_split_fixed_pose_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_focal_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_focal_fixed_point_num_max,
      size_t pinhole_split_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max,
      int device_id = 0);

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
  void SetPinholeCalibNodesFromStackedHost(const double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeCalib nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeCalibNodesFromStackedDevice(const double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedHost(double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedDevice(double* const data,
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
   * Set the current value for the PinholeFocal nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedHost(const double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedDevice(const double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedHost(double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedDevice(double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current number of active nodes of type PinholeFocal.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFocalNum(size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedHost(const double* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedDevice(const double* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedHost(double* const data,
                                        size_t offset,
                                        size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedDevice(double* const data,
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
  void SetPinholePrincipalPointNodesFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current value for the PinholePrincipalPoint nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePrincipalPointNodesFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedHost(double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedDevice(double* const data,
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
  void SetPointNodesFromStackedHost(const double* const data,
                                    size_t offset,
                                    size_t num);

  /**
   * Set the current value for the Point nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedDevice(const double* const data,
                                      size_t offset,
                                      size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedHost(double* const data,
                                  size_t offset,
                                  size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedDevice(double* const data,
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
  void SetSimpleRadialCalibNodesFromStackedHost(const double* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current value for the SimpleRadialCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialCalibNodesFromStackedDevice(const double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedHost(double* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedDevice(double* const data,
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
  void SetSimpleRadialFocalAndExtraNodesFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialFocalAndExtra nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialFocalAndExtraNodesFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndExtra nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndExtraNodesToStackedHost(double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndExtra nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndExtraNodesToStackedDevice(double* const data,
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
  void SetSimpleRadialPoseNodesFromStackedHost(const double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the current value for the SimpleRadialPose nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPoseNodesFromStackedDevice(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedHost(double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedDevice(double* const data,
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
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialPrincipalPoint nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPrincipalPointNodesFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedHost(double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedDevice(double* const data,
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
   * Set the indices for the pose argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialPoseIndicesFromHost(const unsigned int* const indices,
                                          size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialPoseIndicesFromDevice(const unsigned int* const indices,
                                            size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialCalibIndicesFromHost(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialCalibIndicesFromDevice(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialPointIndicesFromHost(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialPointIndicesFromDevice(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadial factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSensorFromRigDataFromStackedHost(const double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadial factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedHost(const double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedDevice(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the current number of SimpleRadial factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialNum(size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPose factor
   * from host.
   */
  void SetSimpleRadialFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPose factor
   * from device.
   */
  void SetSimpleRadialFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialFixedPose factor
   * from host.
   */
  void SetSimpleRadialFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialFixedPose factor
   * from device.
   */
  void SetSimpleRadialFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPose factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPose factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedHost(const double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialFixedPoint factor
   * from host.
   */
  void SetSimpleRadialFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialFixedPoint factor
   * from device.
   */
  void SetSimpleRadialFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPoint
   * factor from host.
   */
  void SetSimpleRadialFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPoint
   * factor from device.
   */
  void SetSimpleRadialFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialFixedPoseFixedPoint factor from host.
   */
  void SetSimpleRadialFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialFixedPoseFixedPoint factor from device.
   */
  void SetSimpleRadialFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialFixedPoseFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialFixedPoseFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the Pinhole factor from host.
   */
  void SetPinholePoseIndicesFromHost(const unsigned int* const indices,
                                     size_t num);

  /**
   * Set the indices for the pose argument for the Pinhole factor from device.
   */
  void SetPinholePoseIndicesFromDevice(const unsigned int* const indices,
                                       size_t num);

  /**
   * Set the indices for the calib argument for the Pinhole factor from host.
   */
  void SetPinholeCalibIndicesFromHost(const unsigned int* const indices,
                                      size_t num);

  /**
   * Set the indices for the calib argument for the Pinhole factor from device.
   */
  void SetPinholeCalibIndicesFromDevice(const unsigned int* const indices,
                                        size_t num);

  /**
   * Set the indices for the point argument for the Pinhole factor from host.
   */
  void SetPinholePointIndicesFromHost(const unsigned int* const indices,
                                      size_t num);

  /**
   * Set the indices for the point argument for the Pinhole factor from device.
   */
  void SetPinholePointIndicesFromDevice(const unsigned int* const indices,
                                        size_t num);

  /**
   * Set the values for the sensor_from_rig consts Pinhole factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSensorFromRigDataFromStackedHost(const double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the values for the sensor_from_rig consts Pinhole factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSensorFromRigDataFromStackedDevice(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedHost(const double* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedDevice(const double* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Set the current number of Pinhole factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPose factor from
   * host.
   */
  void SetPinholeFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPose factor from
   * device.
   */
  void SetPinholeFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPose factor from
   * host.
   */
  void SetPinholeFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPose factor from
   * device.
   */
  void SetPinholeFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedHost(const double* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedDevice(const double* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedHost(const double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedDevice(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current number of PinholeFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPoint factor from
   * host.
   */
  void SetPinholeFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPoint factor from
   * device.
   */
  void SetPinholeFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoint factor
   * from host.
   */
  void SetPinholeFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoint factor
   * from device.
   */
  void SetPinholeFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the current number of PinholeFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoseFixedPoint
   * factor from host.
   */
  void SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoseFixedPoint
   * factor from device.
   */
  void SetPinholeFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from device.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPoseFixedFocalAndExtra
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPoseFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedFocalAndExtraFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the PinholeSplitFixedFocal factor
   * from host.
   */
  void SetPinholeSplitFixedFocalPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeSplitFixedFocal factor
   * from device.
   */
  void SetPinholeSplitFixedFocalPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocal factor from host.
   */
  void SetPinholeSplitFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocal factor from device.
   */
  void SetPinholeSplitFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeSplitFixedFocal
   * factor from host.
   */
  void SetPinholeSplitFixedFocalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeSplitFixedFocal
   * factor from device.
   */
  void SetPinholeSplitFixedFocalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeSplitFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeSplitFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from host.
   */
  void SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from device.
   */
  void SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from host.
   */
  void SetPinholeSplitFixedPoseFixedFocalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from device.
   */
  void SetPinholeSplitFixedPoseFixedFocalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocal factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocal factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalNum(size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocalFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocalFixedPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocalFixedPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocalFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedFocalFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointNum(size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(size_t num);

 private:
  SolverParams<double> params_;
  int device_id_;
  uint8_t* origin_ptr_;
  size_t scratch_inout_size_;
  size_t allocation_size_;

  int solver_iter_;
  int pcg_iter_;

  bool indices_valid_;

  double pcg_r_0_norm2_;
  double pcg_r_kp1_norm2_;

  size_t PinholeCalib_num_;
  size_t PinholeCalib_num_max_;
  size_t PinholeFocal_num_;
  size_t PinholeFocal_num_max_;
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
  size_t simple_radial_num_;
  size_t simple_radial_num_max_;
  size_t simple_radial_fixed_pose_num_;
  size_t simple_radial_fixed_pose_num_max_;
  size_t simple_radial_fixed_point_num_;
  size_t simple_radial_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_point_num_max_;
  size_t pinhole_num_;
  size_t pinhole_num_max_;
  size_t pinhole_fixed_pose_num_;
  size_t pinhole_fixed_pose_num_max_;
  size_t pinhole_fixed_point_num_;
  size_t pinhole_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_point_num_max_;
  size_t simple_radial_split_fixed_focal_and_extra_num_;
  size_t simple_radial_split_fixed_focal_and_extra_num_max_;
  size_t simple_radial_split_fixed_principal_point_num_;
  size_t simple_radial_split_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_focal_and_extra_num_;
  size_t simple_radial_split_fixed_pose_fixed_focal_and_extra_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_num_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_focal_and_extra_fixed_point_num_;
  size_t simple_radial_split_fixed_focal_and_extra_fixed_point_num_max_;
  size_t simple_radial_split_fixed_principal_point_fixed_point_num_;
  size_t simple_radial_split_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_;
  size_t
      simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_focal_num_;
  size_t pinhole_split_fixed_focal_num_max_;
  size_t pinhole_split_fixed_principal_point_num_;
  size_t pinhole_split_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_num_max_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_num_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_num_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_point_num_;
  size_t pinhole_split_fixed_focal_fixed_point_num_max_;
  size_t pinhole_split_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_;

  size_t get_nbytes();
  double LinearizeFirst();
  void Linearize();
  double DoResJacFirst();
  void DoResJac();
  void DoNormalize();
  void DoJtjpDirect();
  void DoAlphaFirst();
  void DoAlpha();
  void DoUpdateStepFirst();
  void DoUpdateStep();
  void DoUpdateRFirst();
  void DoUpdateR();
  double DoRetractScore();
  void DoBeta();
  void DoUpdateP();
  void DoUpdateMp();
  double GetPredDecrease();

  double* marker__start_;
  double* nodes__PinholeCalib__storage_current_;
  double* nodes__PinholeCalib__storage_check_;
  double* nodes__PinholeCalib__storage_new_best_;
  double* nodes__PinholeFocal__storage_current_;
  double* nodes__PinholeFocal__storage_check_;
  double* nodes__PinholeFocal__storage_new_best_;
  double* nodes__PinholePose__storage_current_;
  double* nodes__PinholePose__storage_check_;
  double* nodes__PinholePose__storage_new_best_;
  double* nodes__PinholePrincipalPoint__storage_current_;
  double* nodes__PinholePrincipalPoint__storage_check_;
  double* nodes__PinholePrincipalPoint__storage_new_best_;
  double* nodes__Point__storage_current_;
  double* nodes__Point__storage_check_;
  double* nodes__Point__storage_new_best_;
  double* nodes__SimpleRadialCalib__storage_current_;
  double* nodes__SimpleRadialCalib__storage_check_;
  double* nodes__SimpleRadialCalib__storage_new_best_;
  double* nodes__SimpleRadialFocalAndExtra__storage_current_;
  double* nodes__SimpleRadialFocalAndExtra__storage_check_;
  double* nodes__SimpleRadialFocalAndExtra__storage_new_best_;
  double* nodes__SimpleRadialPose__storage_current_;
  double* nodes__SimpleRadialPose__storage_check_;
  double* nodes__SimpleRadialPose__storage_new_best_;
  double* nodes__SimpleRadialPrincipalPoint__storage_current_;
  double* nodes__SimpleRadialPrincipalPoint__storage_check_;
  double* nodes__SimpleRadialPrincipalPoint__storage_new_best_;
  SharedIndex* facs__simple_radial__args__pose__idx_shared_;
  double* facs__simple_radial__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial__args__point__idx_shared_;
  double* facs__simple_radial__args__pixel__data_;
  double* facs__simple_radial_fixed_pose__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_pose__args__point__idx_shared_;
  double* facs__simple_radial_fixed_pose__args__pixel__data_;
  double* facs__simple_radial_fixed_pose__args__pose__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__pose__idx_shared_;
  double* facs__simple_radial_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__calib__idx_shared_;
  double* facs__simple_radial_fixed_point__args__pixel__data_;
  double* facs__simple_radial_fixed_point__args__point__data_;
  double*
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__pose__data_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole__args__pose__idx_shared_;
  double* facs__pinhole__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole__args__calib__idx_shared_;
  SharedIndex* facs__pinhole__args__point__idx_shared_;
  double* facs__pinhole__args__pixel__data_;
  double* facs__pinhole_fixed_pose__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose__args__point__idx_shared_;
  double* facs__pinhole_fixed_pose__args__pixel__data_;
  double* facs__pinhole_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_fixed_point__args__pose__idx_shared_;
  double* facs__pinhole_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_point__args__calib__idx_shared_;
  double* facs__pinhole_fixed_point__args__pixel__data_;
  double* facs__pinhole_fixed_point__args__point__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_;
  double* facs__pinhole_fixed_pose_fixed_point__args__pixel__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__pose__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_;
  double* facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_;
  double* facs__simple_radial_split_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__pose__idx_shared_;
  double* facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_focal__args__pixel__data_;
  double* facs__pinhole_split_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_;
  double* marker__scratch_inout_;
  double* facs__simple_radial__res_;
  double* facs__simple_radial_fixed_pose__res_;
  double* facs__simple_radial_fixed_point__res_;
  double* facs__simple_radial_fixed_pose_fixed_point__res_;
  double* facs__pinhole__res_;
  double* facs__pinhole_fixed_pose__res_;
  double* facs__pinhole_fixed_point__res_;
  double* facs__pinhole_fixed_pose_fixed_point__res_;
  double* facs__simple_radial_split_fixed_focal_and_extra__res_;
  double* facs__simple_radial_split_fixed_principal_point__res_;
  double* facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__res_;
  double* facs__simple_radial_split_fixed_pose_fixed_principal_point__res_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__res_;
  double* facs__simple_radial_split_fixed_focal_and_extra_fixed_point__res_;
  double* facs__simple_radial_split_fixed_principal_point_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_;
  double* facs__pinhole_split_fixed_focal__res_;
  double* facs__pinhole_split_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__res_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_focal_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_focal_fixed_point__res_;
  double* facs__pinhole_split_fixed_principal_point_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_;
  double* facs__simple_radial__args__pose__jac_;
  double* facs__simple_radial__args__calib__jac_;
  double* facs__simple_radial__args__point__jac_;
  double* facs__simple_radial_fixed_pose__args__calib__jac_;
  double* facs__simple_radial_fixed_pose__args__point__jac_;
  double* facs__simple_radial_fixed_point__args__pose__jac_;
  double* facs__simple_radial_fixed_point__args__calib__jac_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__calib__jac_;
  double* facs__pinhole__args__pose__jac_;
  double* facs__pinhole__args__calib__jac_;
  double* facs__pinhole__args__point__jac_;
  double* facs__pinhole_fixed_pose__args__calib__jac_;
  double* facs__pinhole_fixed_pose__args__point__jac_;
  double* facs__pinhole_fixed_point__args__pose__jac_;
  double* facs__pinhole_fixed_point__args__calib__jac_;
  double* facs__pinhole_fixed_pose_fixed_point__args__calib__jac_;
  double* facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_;
  double* facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_;
  double* facs__simple_radial_split_fixed_principal_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_;
  double* facs__simple_radial_split_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_;
  double* facs__pinhole_split_fixed_focal__args__pose__jac_;
  double* facs__pinhole_split_fixed_focal__args__principal_point__jac_;
  double* facs__pinhole_split_fixed_focal__args__point__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__pose__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__focal__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__jac_;
  double* nodes__PinholeCalib__z_;
  double* nodes__PinholeCalib__z_end__;
  double* nodes__PinholeFocal__z_;
  double* nodes__PinholeFocal__z_end__;
  double* nodes__PinholePose__z_;
  double* nodes__PinholePose__z_end__;
  double* nodes__PinholePrincipalPoint__z_;
  double* nodes__PinholePrincipalPoint__z_end__;
  double* nodes__Point__z_;
  double* nodes__Point__z_end__;
  double* nodes__SimpleRadialCalib__z_;
  double* nodes__SimpleRadialCalib__z_end__;
  double* nodes__SimpleRadialFocalAndExtra__z_;
  double* nodes__SimpleRadialFocalAndExtra__z_end__;
  double* nodes__SimpleRadialPose__z_;
  double* nodes__SimpleRadialPose__z_end__;
  double* nodes__SimpleRadialPrincipalPoint__z_;
  double* nodes__SimpleRadialPrincipalPoint__z_end__;
  double* nodes__PinholeCalib__p_;
  double* nodes__PinholeCalib__p_end__;
  double* nodes__PinholeFocal__p_;
  double* nodes__PinholeFocal__p_end__;
  double* nodes__PinholePose__p_;
  double* nodes__PinholePose__p_end__;
  double* nodes__PinholePrincipalPoint__p_;
  double* nodes__PinholePrincipalPoint__p_end__;
  double* nodes__Point__p_;
  double* nodes__Point__p_end__;
  double* nodes__SimpleRadialCalib__p_;
  double* nodes__SimpleRadialCalib__p_end__;
  double* nodes__SimpleRadialFocalAndExtra__p_;
  double* nodes__SimpleRadialFocalAndExtra__p_end__;
  double* nodes__SimpleRadialPose__p_;
  double* nodes__SimpleRadialPose__p_end__;
  double* nodes__SimpleRadialPrincipalPoint__p_;
  double* nodes__SimpleRadialPrincipalPoint__p_end__;
  double* nodes__PinholeCalib__step_;
  double* nodes__PinholeCalib__step_end__;
  double* nodes__PinholeFocal__step_;
  double* nodes__PinholeFocal__step_end__;
  double* nodes__PinholePose__step_;
  double* nodes__PinholePose__step_end__;
  double* nodes__PinholePrincipalPoint__step_;
  double* nodes__PinholePrincipalPoint__step_end__;
  double* nodes__Point__step_;
  double* nodes__Point__step_end__;
  double* nodes__SimpleRadialCalib__step_;
  double* nodes__SimpleRadialCalib__step_end__;
  double* nodes__SimpleRadialFocalAndExtra__step_;
  double* nodes__SimpleRadialFocalAndExtra__step_end__;
  double* nodes__SimpleRadialPose__step_;
  double* nodes__SimpleRadialPose__step_end__;
  double* nodes__SimpleRadialPrincipalPoint__step_;
  double* nodes__SimpleRadialPrincipalPoint__step_end__;
  double* marker__w_start_;
  double* nodes__PinholeCalib__w_;
  double* nodes__PinholeFocal__w_;
  double* nodes__PinholePose__w_;
  double* nodes__PinholePrincipalPoint__w_;
  double* nodes__Point__w_;
  double* nodes__SimpleRadialCalib__w_;
  double* nodes__SimpleRadialFocalAndExtra__w_;
  double* nodes__SimpleRadialPose__w_;
  double* nodes__SimpleRadialPrincipalPoint__w_;
  double* marker__w_end_;
  double* marker__r_0_start_;
  double* nodes__PinholeCalib__r_0_;
  double* nodes__PinholeFocal__r_0_;
  double* nodes__PinholePose__r_0_;
  double* nodes__PinholePrincipalPoint__r_0_;
  double* nodes__Point__r_0_;
  double* nodes__SimpleRadialCalib__r_0_;
  double* nodes__SimpleRadialFocalAndExtra__r_0_;
  double* nodes__SimpleRadialPose__r_0_;
  double* nodes__SimpleRadialPrincipalPoint__r_0_;
  double* marker__r_0_end_;
  double* marker__r_k_start_;
  double* nodes__PinholeCalib__r_k_;
  double* nodes__PinholeFocal__r_k_;
  double* nodes__PinholePose__r_k_;
  double* nodes__PinholePrincipalPoint__r_k_;
  double* nodes__Point__r_k_;
  double* nodes__SimpleRadialCalib__r_k_;
  double* nodes__SimpleRadialFocalAndExtra__r_k_;
  double* nodes__SimpleRadialPose__r_k_;
  double* nodes__SimpleRadialPrincipalPoint__r_k_;
  double* marker__r_k_end_;
  double* marker__Mp_start_;
  double* nodes__PinholeCalib__Mp_;
  double* nodes__PinholeFocal__Mp_;
  double* nodes__PinholePose__Mp_;
  double* nodes__PinholePrincipalPoint__Mp_;
  double* nodes__Point__Mp_;
  double* nodes__SimpleRadialCalib__Mp_;
  double* nodes__SimpleRadialFocalAndExtra__Mp_;
  double* nodes__SimpleRadialPose__Mp_;
  double* nodes__SimpleRadialPrincipalPoint__Mp_;
  double* marker__Mp_end_;
  double* marker__precond_start_;
  double* nodes__PinholeCalib__precond_diag_;
  double* nodes__PinholeCalib__precond_tril_;
  double* nodes__PinholeFocal__precond_diag_;
  double* nodes__PinholeFocal__precond_tril_;
  double* nodes__PinholePose__precond_diag_;
  double* nodes__PinholePose__precond_tril_;
  double* nodes__PinholePrincipalPoint__precond_diag_;
  double* nodes__PinholePrincipalPoint__precond_tril_;
  double* nodes__Point__precond_diag_;
  double* nodes__Point__precond_tril_;
  double* nodes__SimpleRadialCalib__precond_diag_;
  double* nodes__SimpleRadialCalib__precond_tril_;
  double* nodes__SimpleRadialFocalAndExtra__precond_diag_;
  double* nodes__SimpleRadialFocalAndExtra__precond_tril_;
  double* nodes__SimpleRadialPose__precond_diag_;
  double* nodes__SimpleRadialPose__precond_tril_;
  double* nodes__SimpleRadialPrincipalPoint__precond_diag_;
  double* nodes__SimpleRadialPrincipalPoint__precond_tril_;
  double* marker__precond_end_;
  double* marker__jp_start_;
  double* facs__simple_radial__jp_;
  double* facs__simple_radial_fixed_pose__jp_;
  double* facs__simple_radial_fixed_point__jp_;
  double* facs__simple_radial_fixed_pose_fixed_point__jp_;
  double* facs__pinhole__jp_;
  double* facs__pinhole_fixed_pose__jp_;
  double* facs__pinhole_fixed_point__jp_;
  double* facs__pinhole_fixed_pose_fixed_point__jp_;
  double* facs__simple_radial_split_fixed_focal_and_extra__jp_;
  double* facs__simple_radial_split_fixed_principal_point__jp_;
  double* facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__jp_;
  double* facs__simple_radial_split_fixed_pose_fixed_principal_point__jp_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__jp_;
  double* facs__simple_radial_split_fixed_focal_and_extra_fixed_point__jp_;
  double* facs__simple_radial_split_fixed_principal_point_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_fixed_focal__jp_;
  double* facs__pinhole_split_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_focal_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_focal_fixed_point__jp_;
  double* facs__pinhole_split_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__jp_;
  double* marker__jp_end_;
  double* solver__current_diag_;
  double* solver__alpha_numerator_;
  double* solver__alpha_denominator_;
  double* solver__alpha_;
  double* solver__neg_alpha_;
  double* solver__beta_numerator_;
  double* solver__beta_;
  double* solver__r_0_norm2_tot_;
  double* solver__r_kp1_norm2_tot_;
  double* solver__pred_decrease_tot_;
  double* solver__res_tot_;
};

}  // namespace caspar