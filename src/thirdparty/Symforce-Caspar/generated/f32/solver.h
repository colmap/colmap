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
   * @param ThinPrismFisheyeCalib_num_max the maximum number of
   * ThinPrismFisheyeCalibs
   * @param ThinPrismFisheyeFocalAndExtra_num_max the maximum number of
   * ThinPrismFisheyeFocalAndExtras
   * @param ThinPrismFisheyePose_num_max the maximum number of
   * ThinPrismFisheyePoses
   * @param ThinPrismFisheyePrincipalPoint_num_max the maximum number of
   * ThinPrismFisheyePrincipalPoints
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
   * @param thin_prism_fisheye_num_max the maximum number of thin_prism_fisheyes
   * @param thin_prism_fisheye_fixed_pose_num_max the maximum number of
   * thin_prism_fisheye_fixed_poses
   * @param thin_prism_fisheye_fixed_point_num_max the maximum number of
   * thin_prism_fisheye_fixed_points
   * @param thin_prism_fisheye_fixed_pose_fixed_point_num_max the maximum number
   * of thin_prism_fisheye_fixed_pose_fixed_points
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
   * @param thin_prism_fisheye_split_fixed_focal_and_extra_num_max the maximum
   * number of thin_prism_fisheye_split_fixed_focal_and_extras
   * @param thin_prism_fisheye_split_fixed_principal_point_num_max the maximum
   * number of thin_prism_fisheye_split_fixed_principal_points
   * @param thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extras
   * @param thin_prism_fisheye_split_fixed_pose_fixed_principal_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_pose_fixed_principal_points
   * @param
   * thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_points
   * @param thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_focal_and_extra_fixed_points
   * @param thin_prism_fisheye_split_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_principal_point_fixed_points
   * @param
   * thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_points
   * @param
   * thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_points
   * @param
   * thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_points
   * @param
   * thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_points
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
      size_t ThinPrismFisheyeCalib_num_max,
      size_t ThinPrismFisheyeFocalAndExtra_num_max,
      size_t ThinPrismFisheyePose_num_max,
      size_t ThinPrismFisheyePrincipalPoint_num_max,
      size_t simple_radial_num_max,
      size_t simple_radial_fixed_pose_num_max,
      size_t simple_radial_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_point_num_max,
      size_t pinhole_num_max,
      size_t pinhole_fixed_pose_num_max,
      size_t pinhole_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_point_num_max,
      size_t thin_prism_fisheye_num_max,
      size_t thin_prism_fisheye_fixed_pose_num_max,
      size_t thin_prism_fisheye_fixed_point_num_max,
      size_t thin_prism_fisheye_fixed_pose_fixed_point_num_max,
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
      size_t thin_prism_fisheye_split_fixed_focal_and_extra_num_max,
      size_t thin_prism_fisheye_split_fixed_principal_point_num_max,
      size_t thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_num_max,
      size_t thin_prism_fisheye_split_fixed_pose_fixed_principal_point_num_max,
      size_t
          thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_num_max,
      size_t thin_prism_fisheye_split_fixed_principal_point_fixed_point_num_max,
      size_t
          thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max,
      size_t
          thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max,
      size_t
          thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max,
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
   * Set the current value for the PinholeFocal nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedHost(const float* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedDevice(const float* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedHost(float* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedDevice(float* const data,
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
   * Set the current value for the ThinPrismFisheyeCalib nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyeCalibNodesFromStackedHost(const float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current value for the ThinPrismFisheyeCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyeCalibNodesFromStackedDevice(const float* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the ThinPrismFisheyeCalib nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyeCalibNodesToStackedHost(float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the ThinPrismFisheyeCalib nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyeCalibNodesToStackedDevice(float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current number of active nodes of type ThinPrismFisheyeCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeCalibNum(size_t num);

  /**
   * Set the current value for the ThinPrismFisheyeFocalAndExtra nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyeFocalAndExtraNodesFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current value for the ThinPrismFisheyeFocalAndExtra nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyeFocalAndExtraNodesFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ThinPrismFisheyeFocalAndExtra nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyeFocalAndExtraNodesToStackedHost(float* const data,
                                                          size_t offset,
                                                          size_t num);

  /**
   * Read the current value for the ThinPrismFisheyeFocalAndExtra nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyeFocalAndExtraNodesToStackedDevice(float* const data,
                                                            size_t offset,
                                                            size_t num);

  /**
   * Set the current number of active nodes of type
   * ThinPrismFisheyeFocalAndExtra.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeFocalAndExtraNum(size_t num);

  /**
   * Set the current value for the ThinPrismFisheyePose nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyePoseNodesFromStackedHost(const float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the current value for the ThinPrismFisheyePose nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyePoseNodesFromStackedDevice(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Read the current value for the ThinPrismFisheyePose nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyePoseNodesToStackedHost(float* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Read the current value for the ThinPrismFisheyePose nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyePoseNodesToStackedDevice(float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the current number of active nodes of type ThinPrismFisheyePose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyePoseNum(size_t num);

  /**
   * Set the current value for the ThinPrismFisheyePrincipalPoint nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyePrincipalPointNodesFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current value for the ThinPrismFisheyePrincipalPoint nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetThinPrismFisheyePrincipalPointNodesFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Read the current value for the ThinPrismFisheyePrincipalPoint nodes into
   * the stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyePrincipalPointNodesToStackedHost(float* const data,
                                                           size_t offset,
                                                           size_t num);

  /**
   * Read the current value for the ThinPrismFisheyePrincipalPoint nodes into
   * the stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetThinPrismFisheyePrincipalPointNodesToStackedDevice(float* const data,
                                                             size_t offset,
                                                             size_t num);

  /**
   * Set the current number of active nodes of type
   * ThinPrismFisheyePrincipalPoint.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyePrincipalPointNum(size_t num);

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
  void SetSimpleRadialSensorFromRigDataFromStackedHost(const float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadial factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedHost(const float* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedDevice(const float* const data,
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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPose factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedHost(const float* const data,
                                                        size_t offset,
                                                        size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedHost(const float* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts SimpleRadialFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialFixedPoseFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
  void SetPinholeSensorFromRigDataFromStackedHost(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the values for the sensor_from_rig consts Pinhole factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSensorFromRigDataFromStackedDevice(const float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedHost(const float* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedDevice(const float* const data,
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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedHost(const float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedDevice(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedHost(const float* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedDevice(const float* const data,
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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedHost(const float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedDevice(const float* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedHost(const float* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedDevice(const float* const data,
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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the ThinPrismFisheye factor from
   * host.
   */
  void SetThinPrismFisheyePoseIndicesFromHost(const unsigned int* const indices,
                                              size_t num);

  /**
   * Set the indices for the pose argument for the ThinPrismFisheye factor from
   * device.
   */
  void SetThinPrismFisheyePoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheye factor from
   * host.
   */
  void SetThinPrismFisheyeCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheye factor from
   * device.
   */
  void SetThinPrismFisheyeCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the ThinPrismFisheye factor from
   * host.
   */
  void SetThinPrismFisheyePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the ThinPrismFisheye factor from
   * device.
   */
  void SetThinPrismFisheyePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheye factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheye factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheye factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyePixelDataFromStackedHost(const float* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheye factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyePixelDataFromStackedDevice(const float* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the current number of ThinPrismFisheye factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeNum(size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheyeFixedPose
   * factor from host.
   */
  void SetThinPrismFisheyeFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheyeFixedPose
   * factor from device.
   */
  void SetThinPrismFisheyeFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the ThinPrismFisheyeFixedPose
   * factor from host.
   */
  void SetThinPrismFisheyeFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the ThinPrismFisheyeFixedPose
   * factor from device.
   */
  void SetThinPrismFisheyeFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheyeFixedPose
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheyeFixedPose
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPosePixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPosePixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts ThinPrismFisheyeFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPosePoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts ThinPrismFisheyeFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPosePoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the ThinPrismFisheyeFixedPoint
   * factor from host.
   */
  void SetThinPrismFisheyeFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the ThinPrismFisheyeFixedPoint
   * factor from device.
   */
  void SetThinPrismFisheyeFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheyeFixedPoint
   * factor from host.
   */
  void SetThinPrismFisheyeFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the ThinPrismFisheyeFixedPoint
   * factor from device.
   */
  void SetThinPrismFisheyeFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheyeFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts ThinPrismFisheyeFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts ThinPrismFisheyeFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts ThinPrismFisheyeFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the
   * ThinPrismFisheyeFixedPoseFixedPoint factor from host.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * ThinPrismFisheyeFixedPoseFixedPoint factor from device.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeFixedPoseFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeFixedPoseFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts ThinPrismFisheyeFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeFixedPoseFixedPointNum(size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts PinholeSplitFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocal factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

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
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from host.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from device.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from host.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from host.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from device.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeSplitFixedFocalAndExtra
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts ThinPrismFisheyeSplitFixedFocalAndExtra
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtra factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeSplitFixedFocalAndExtra factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from device.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from device.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeSplitFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from host.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtra
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraNum(size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeSplitFixedPoseFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * host.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * host.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from host.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of ThinPrismFisheyeSplitFixedFocalAndExtraFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from host.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * host.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPointNum(
      size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * host.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_extra argument for the
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * device.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from host.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from device.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the sensor_from_rig consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointSensorFromRigDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_extra consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const float* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointNum(
      size_t num);

 private:
  SolverParams<float> params_;
  int device_id_;
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
  size_t ThinPrismFisheyeCalib_num_;
  size_t ThinPrismFisheyeCalib_num_max_;
  size_t ThinPrismFisheyeFocalAndExtra_num_;
  size_t ThinPrismFisheyeFocalAndExtra_num_max_;
  size_t ThinPrismFisheyePose_num_;
  size_t ThinPrismFisheyePose_num_max_;
  size_t ThinPrismFisheyePrincipalPoint_num_;
  size_t ThinPrismFisheyePrincipalPoint_num_max_;
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
  size_t thin_prism_fisheye_num_;
  size_t thin_prism_fisheye_num_max_;
  size_t thin_prism_fisheye_fixed_pose_num_;
  size_t thin_prism_fisheye_fixed_pose_num_max_;
  size_t thin_prism_fisheye_fixed_point_num_;
  size_t thin_prism_fisheye_fixed_point_num_max_;
  size_t thin_prism_fisheye_fixed_pose_fixed_point_num_;
  size_t thin_prism_fisheye_fixed_pose_fixed_point_num_max_;
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
  size_t thin_prism_fisheye_split_fixed_focal_and_extra_num_;
  size_t thin_prism_fisheye_split_fixed_focal_and_extra_num_max_;
  size_t thin_prism_fisheye_split_fixed_principal_point_num_;
  size_t thin_prism_fisheye_split_fixed_principal_point_num_max_;
  size_t thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_num_;
  size_t thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_num_max_;
  size_t thin_prism_fisheye_split_fixed_pose_fixed_principal_point_num_;
  size_t thin_prism_fisheye_split_fixed_pose_fixed_principal_point_num_max_;
  size_t
      thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_num_;
  size_t thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_num_max_;
  size_t thin_prism_fisheye_split_fixed_principal_point_fixed_point_num_;
  size_t thin_prism_fisheye_split_fixed_principal_point_fixed_point_num_max_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point_num_max_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t
      thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t
      thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_;
  size_t
      thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max_;

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
  float* nodes__PinholeFocal__storage_current_;
  float* nodes__PinholeFocal__storage_check_;
  float* nodes__PinholeFocal__storage_new_best_;
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
  float* nodes__ThinPrismFisheyeCalib__storage_current_;
  float* nodes__ThinPrismFisheyeCalib__storage_check_;
  float* nodes__ThinPrismFisheyeCalib__storage_new_best_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__storage_current_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__storage_check_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__storage_new_best_;
  float* nodes__ThinPrismFisheyePose__storage_current_;
  float* nodes__ThinPrismFisheyePose__storage_check_;
  float* nodes__ThinPrismFisheyePose__storage_new_best_;
  float* nodes__ThinPrismFisheyePrincipalPoint__storage_current_;
  float* nodes__ThinPrismFisheyePrincipalPoint__storage_check_;
  float* nodes__ThinPrismFisheyePrincipalPoint__storage_new_best_;
  SharedIndex* facs__simple_radial__args__pose__idx_shared_;
  float* facs__simple_radial__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial__args__point__idx_shared_;
  float* facs__simple_radial__args__pixel__data_;
  float* facs__simple_radial_fixed_pose__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_pose__args__point__idx_shared_;
  float* facs__simple_radial_fixed_pose__args__pixel__data_;
  float* facs__simple_radial_fixed_pose__args__pose__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__pose__idx_shared_;
  float* facs__simple_radial_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__calib__idx_shared_;
  float* facs__simple_radial_fixed_point__args__pixel__data_;
  float* facs__simple_radial_fixed_point__args__point__data_;
  float*
      facs__simple_radial_fixed_pose_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__pose__data_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole__args__pose__idx_shared_;
  float* facs__pinhole__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole__args__calib__idx_shared_;
  SharedIndex* facs__pinhole__args__point__idx_shared_;
  float* facs__pinhole__args__pixel__data_;
  float* facs__pinhole_fixed_pose__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose__args__point__idx_shared_;
  float* facs__pinhole_fixed_pose__args__pixel__data_;
  float* facs__pinhole_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_fixed_point__args__pose__idx_shared_;
  float* facs__pinhole_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_point__args__calib__idx_shared_;
  float* facs__pinhole_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_point__args__point__data_;
  float* facs__pinhole_fixed_pose_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_;
  float* facs__pinhole_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__pinhole_fixed_pose_fixed_point__args__pose__data_;
  float* facs__pinhole_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__thin_prism_fisheye__args__pose__idx_shared_;
  float* facs__thin_prism_fisheye__args__sensor_from_rig__data_;
  SharedIndex* facs__thin_prism_fisheye__args__calib__idx_shared_;
  SharedIndex* facs__thin_prism_fisheye__args__point__idx_shared_;
  float* facs__thin_prism_fisheye__args__pixel__data_;
  float* facs__thin_prism_fisheye_fixed_pose__args__sensor_from_rig__data_;
  SharedIndex* facs__thin_prism_fisheye_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__thin_prism_fisheye_fixed_pose__args__point__idx_shared_;
  float* facs__thin_prism_fisheye_fixed_pose__args__pixel__data_;
  float* facs__thin_prism_fisheye_fixed_pose__args__pose__data_;
  SharedIndex* facs__thin_prism_fisheye_fixed_point__args__pose__idx_shared_;
  float* facs__thin_prism_fisheye_fixed_point__args__sensor_from_rig__data_;
  SharedIndex* facs__thin_prism_fisheye_fixed_point__args__calib__idx_shared_;
  float* facs__thin_prism_fisheye_fixed_point__args__pixel__data_;
  float* facs__thin_prism_fisheye_fixed_point__args__point__data_;
  float*
      facs__thin_prism_fisheye_fixed_pose_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_fixed_pose_fixed_point__args__calib__idx_shared_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__args__pixel__data_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__args__pose__data_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra__args__point__idx_shared_;
  float* facs__simple_radial_split_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_;
  float* facs__simple_radial_split_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__pose__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__pose__idx_shared_;
  float* facs__pinhole_split_fixed_focal__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__point__idx_shared_;
  float* facs__pinhole_split_fixed_focal__args__pixel__data_;
  float* facs__pinhole_split_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__pinhole_split_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_;
  float* facs__pinhole_split_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_;
  float*
      facs__pinhole_split_fixed_focal_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  float* facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_;
  float* facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_;
  float* facs__pinhole_split_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__focal_and_extra__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__principal_point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__principal_point__idx_shared_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__pose__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__focal_and_extra__idx_shared_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__pose__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__point__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__pose__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__principal_point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__pose__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__sensor_from_rig__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pixel__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__focal_and_extra__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__principal_point__data_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__point__data_;
  float* marker__scratch_inout_;
  float* facs__simple_radial__res_;
  float* facs__simple_radial_fixed_pose__res_;
  float* facs__simple_radial_fixed_point__res_;
  float* facs__simple_radial_fixed_pose_fixed_point__res_;
  float* facs__pinhole__res_;
  float* facs__pinhole_fixed_pose__res_;
  float* facs__pinhole_fixed_point__res_;
  float* facs__pinhole_fixed_pose_fixed_point__res_;
  float* facs__thin_prism_fisheye__res_;
  float* facs__thin_prism_fisheye_fixed_pose__res_;
  float* facs__thin_prism_fisheye_fixed_point__res_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__res_;
  float* facs__simple_radial_split_fixed_focal_and_extra__res_;
  float* facs__simple_radial_split_fixed_principal_point__res_;
  float* facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__res_;
  float* facs__simple_radial_split_fixed_pose_fixed_principal_point__res_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__simple_radial_split_fixed_focal_and_extra_fixed_point__res_;
  float* facs__simple_radial_split_fixed_principal_point_fixed_point__res_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_;
  float* facs__pinhole_split_fixed_focal__res_;
  float* facs__pinhole_split_fixed_principal_point__res_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__res_;
  float* facs__pinhole_split_fixed_pose_fixed_principal_point__res_;
  float* facs__pinhole_split_fixed_focal_fixed_principal_point__res_;
  float* facs__pinhole_split_fixed_focal_fixed_point__res_;
  float* facs__pinhole_split_fixed_principal_point_fixed_point__res_;
  float* facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_;
  float* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_;
  float* facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_;
  float* facs__thin_prism_fisheye_split_fixed_focal_and_extra__res_;
  float* facs__thin_prism_fisheye_split_fixed_principal_point__res_;
  float* facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__res_;
  float* facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__res_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__res_;
  float* facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__res_;
  float* facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__res_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__res_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__res_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__res_;
  float* facs__simple_radial__args__pose__jac_;
  float* facs__simple_radial__args__calib__jac_;
  float* facs__simple_radial__args__point__jac_;
  float* facs__simple_radial_fixed_pose__args__calib__jac_;
  float* facs__simple_radial_fixed_pose__args__point__jac_;
  float* facs__simple_radial_fixed_point__args__pose__jac_;
  float* facs__simple_radial_fixed_point__args__calib__jac_;
  float* facs__simple_radial_fixed_pose_fixed_point__args__calib__jac_;
  float* facs__pinhole__args__pose__jac_;
  float* facs__pinhole__args__calib__jac_;
  float* facs__pinhole__args__point__jac_;
  float* facs__pinhole_fixed_pose__args__calib__jac_;
  float* facs__pinhole_fixed_pose__args__point__jac_;
  float* facs__pinhole_fixed_point__args__pose__jac_;
  float* facs__pinhole_fixed_point__args__calib__jac_;
  float* facs__pinhole_fixed_pose_fixed_point__args__calib__jac_;
  float* facs__thin_prism_fisheye__args__pose__jac_;
  float* facs__thin_prism_fisheye__args__calib__jac_;
  float* facs__thin_prism_fisheye__args__point__jac_;
  float* facs__thin_prism_fisheye_fixed_pose__args__calib__jac_;
  float* facs__thin_prism_fisheye_fixed_pose__args__point__jac_;
  float* facs__thin_prism_fisheye_fixed_point__args__pose__jac_;
  float* facs__thin_prism_fisheye_fixed_point__args__calib__jac_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__args__calib__jac_;
  float* facs__simple_radial_split_fixed_focal_and_extra__args__pose__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra__args__principal_point__jac_;
  float* facs__simple_radial_split_fixed_focal_and_extra__args__point__jac_;
  float* facs__simple_radial_split_fixed_principal_point__args__pose__jac_;
  float*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_extra__jac_;
  float* facs__simple_radial_split_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__args__point__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__pose__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_;
  float*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_;
  float* facs__pinhole_split_fixed_focal__args__pose__jac_;
  float* facs__pinhole_split_fixed_focal__args__principal_point__jac_;
  float* facs__pinhole_split_fixed_focal__args__point__jac_;
  float* facs__pinhole_split_fixed_principal_point__args__pose__jac_;
  float* facs__pinhole_split_fixed_principal_point__args__focal__jac_;
  float* facs__pinhole_split_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_;
  float* facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_;
  float*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_;
  float*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__jac_;
  float*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__jac_;
  float*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__jac_;
  float* facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__pose__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__principal_point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra__args__point__jac_;
  float* facs__thin_prism_fisheye_split_fixed_principal_point__args__pose__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__focal_and_extra__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point__args__point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__principal_point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__args__point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__focal_and_extra__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__args__point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__pose__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__pose__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__pose__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__args__point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__args__principal_point__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_extra__jac_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__args__pose__jac_;
  float* nodes__PinholeCalib__z_;
  float* nodes__PinholeCalib__z_end__;
  float* nodes__PinholeFocal__z_;
  float* nodes__PinholeFocal__z_end__;
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
  float* nodes__ThinPrismFisheyeCalib__z_;
  float* nodes__ThinPrismFisheyeCalib__z_end__;
  float* nodes__ThinPrismFisheyeFocalAndExtra__z_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__z_end__;
  float* nodes__ThinPrismFisheyePose__z_;
  float* nodes__ThinPrismFisheyePose__z_end__;
  float* nodes__ThinPrismFisheyePrincipalPoint__z_;
  float* nodes__ThinPrismFisheyePrincipalPoint__z_end__;
  float* nodes__PinholeCalib__p_;
  float* nodes__PinholeCalib__p_end__;
  float* nodes__PinholeFocal__p_;
  float* nodes__PinholeFocal__p_end__;
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
  float* nodes__ThinPrismFisheyeCalib__p_;
  float* nodes__ThinPrismFisheyeCalib__p_end__;
  float* nodes__ThinPrismFisheyeFocalAndExtra__p_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__p_end__;
  float* nodes__ThinPrismFisheyePose__p_;
  float* nodes__ThinPrismFisheyePose__p_end__;
  float* nodes__ThinPrismFisheyePrincipalPoint__p_;
  float* nodes__ThinPrismFisheyePrincipalPoint__p_end__;
  float* nodes__PinholeCalib__step_;
  float* nodes__PinholeCalib__step_end__;
  float* nodes__PinholeFocal__step_;
  float* nodes__PinholeFocal__step_end__;
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
  float* nodes__ThinPrismFisheyeCalib__step_;
  float* nodes__ThinPrismFisheyeCalib__step_end__;
  float* nodes__ThinPrismFisheyeFocalAndExtra__step_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__step_end__;
  float* nodes__ThinPrismFisheyePose__step_;
  float* nodes__ThinPrismFisheyePose__step_end__;
  float* nodes__ThinPrismFisheyePrincipalPoint__step_;
  float* nodes__ThinPrismFisheyePrincipalPoint__step_end__;
  float* marker__w_start_;
  float* nodes__PinholeCalib__w_;
  float* nodes__PinholeFocal__w_;
  float* nodes__PinholePose__w_;
  float* nodes__PinholePrincipalPoint__w_;
  float* nodes__Point__w_;
  float* nodes__SimpleRadialCalib__w_;
  float* nodes__SimpleRadialFocalAndExtra__w_;
  float* nodes__SimpleRadialPose__w_;
  float* nodes__SimpleRadialPrincipalPoint__w_;
  float* nodes__ThinPrismFisheyeCalib__w_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__w_;
  float* nodes__ThinPrismFisheyePose__w_;
  float* nodes__ThinPrismFisheyePrincipalPoint__w_;
  float* marker__w_end_;
  float* marker__r_0_start_;
  float* nodes__PinholeCalib__r_0_;
  float* nodes__PinholeFocal__r_0_;
  float* nodes__PinholePose__r_0_;
  float* nodes__PinholePrincipalPoint__r_0_;
  float* nodes__Point__r_0_;
  float* nodes__SimpleRadialCalib__r_0_;
  float* nodes__SimpleRadialFocalAndExtra__r_0_;
  float* nodes__SimpleRadialPose__r_0_;
  float* nodes__SimpleRadialPrincipalPoint__r_0_;
  float* nodes__ThinPrismFisheyeCalib__r_0_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__r_0_;
  float* nodes__ThinPrismFisheyePose__r_0_;
  float* nodes__ThinPrismFisheyePrincipalPoint__r_0_;
  float* marker__r_0_end_;
  float* marker__r_k_start_;
  float* nodes__PinholeCalib__r_k_;
  float* nodes__PinholeFocal__r_k_;
  float* nodes__PinholePose__r_k_;
  float* nodes__PinholePrincipalPoint__r_k_;
  float* nodes__Point__r_k_;
  float* nodes__SimpleRadialCalib__r_k_;
  float* nodes__SimpleRadialFocalAndExtra__r_k_;
  float* nodes__SimpleRadialPose__r_k_;
  float* nodes__SimpleRadialPrincipalPoint__r_k_;
  float* nodes__ThinPrismFisheyeCalib__r_k_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__r_k_;
  float* nodes__ThinPrismFisheyePose__r_k_;
  float* nodes__ThinPrismFisheyePrincipalPoint__r_k_;
  float* marker__r_k_end_;
  float* marker__Mp_start_;
  float* nodes__PinholeCalib__Mp_;
  float* nodes__PinholeFocal__Mp_;
  float* nodes__PinholePose__Mp_;
  float* nodes__PinholePrincipalPoint__Mp_;
  float* nodes__Point__Mp_;
  float* nodes__SimpleRadialCalib__Mp_;
  float* nodes__SimpleRadialFocalAndExtra__Mp_;
  float* nodes__SimpleRadialPose__Mp_;
  float* nodes__SimpleRadialPrincipalPoint__Mp_;
  float* nodes__ThinPrismFisheyeCalib__Mp_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__Mp_;
  float* nodes__ThinPrismFisheyePose__Mp_;
  float* nodes__ThinPrismFisheyePrincipalPoint__Mp_;
  float* marker__Mp_end_;
  float* marker__precond_start_;
  float* nodes__PinholeCalib__precond_diag_;
  float* nodes__PinholeCalib__precond_tril_;
  float* nodes__PinholeFocal__precond_diag_;
  float* nodes__PinholeFocal__precond_tril_;
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
  float* nodes__ThinPrismFisheyeCalib__precond_diag_;
  float* nodes__ThinPrismFisheyeCalib__precond_tril_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__precond_diag_;
  float* nodes__ThinPrismFisheyeFocalAndExtra__precond_tril_;
  float* nodes__ThinPrismFisheyePose__precond_diag_;
  float* nodes__ThinPrismFisheyePose__precond_tril_;
  float* nodes__ThinPrismFisheyePrincipalPoint__precond_diag_;
  float* nodes__ThinPrismFisheyePrincipalPoint__precond_tril_;
  float* marker__precond_end_;
  float* marker__jp_start_;
  float* facs__simple_radial__jp_;
  float* facs__simple_radial_fixed_pose__jp_;
  float* facs__simple_radial_fixed_point__jp_;
  float* facs__simple_radial_fixed_pose_fixed_point__jp_;
  float* facs__pinhole__jp_;
  float* facs__pinhole_fixed_pose__jp_;
  float* facs__pinhole_fixed_point__jp_;
  float* facs__pinhole_fixed_pose_fixed_point__jp_;
  float* facs__thin_prism_fisheye__jp_;
  float* facs__thin_prism_fisheye_fixed_pose__jp_;
  float* facs__thin_prism_fisheye_fixed_point__jp_;
  float* facs__thin_prism_fisheye_fixed_pose_fixed_point__jp_;
  float* facs__simple_radial_split_fixed_focal_and_extra__jp_;
  float* facs__simple_radial_split_fixed_principal_point__jp_;
  float* facs__simple_radial_split_fixed_pose_fixed_focal_and_extra__jp_;
  float* facs__simple_radial_split_fixed_pose_fixed_principal_point__jp_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__simple_radial_split_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__simple_radial_split_fixed_principal_point_fixed_point__jp_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_point__jp_;
  float*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  float*
      facs__simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_;
  float* facs__pinhole_split_fixed_focal__jp_;
  float* facs__pinhole_split_fixed_principal_point__jp_;
  float* facs__pinhole_split_fixed_pose_fixed_focal__jp_;
  float* facs__pinhole_split_fixed_pose_fixed_principal_point__jp_;
  float* facs__pinhole_split_fixed_focal_fixed_principal_point__jp_;
  float* facs__pinhole_split_fixed_focal_fixed_point__jp_;
  float* facs__pinhole_split_fixed_principal_point_fixed_point__jp_;
  float* facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__jp_;
  float* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__jp_;
  float* facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  float* facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__jp_;
  float* facs__thin_prism_fisheye_split_fixed_focal_and_extra__jp_;
  float* facs__thin_prism_fisheye_split_fixed_principal_point__jp_;
  float* facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra__jp_;
  float* facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point__jp_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point__jp_;
  float* facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point__jp_;
  float* facs__thin_prism_fisheye_split_fixed_principal_point_fixed_point__jp_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point__jp_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_point__jp_;
  float*
      facs__thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  float*
      facs__thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point__jp_;
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