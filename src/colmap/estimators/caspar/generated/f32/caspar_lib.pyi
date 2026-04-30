from __future__ import annotations

import typing as T

"""
Any object with a valid __array_interface__

https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface
"""
Array = T.Any

"""
Any object with a valid __cuda_array_interface__

https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface
"""
CudaArray = T.Any

class ExitReason:
    MAX_ITERATIONS: int
    CONVERGED_SCORE_THRESHOLD: int
    CONVERGED_DIAG_EXIT: int

class IterationData:
    solver_iter: int
    pcg_iter: int
    score_current: float
    score_best: float
    step_quality: float
    diag: float
    dt_inc: float
    dt_tot: float
    step_accepted: bool

class SolveResult:
    initial_score: float
    final_score: float
    iteration_count: int
    runtime: float
    exit_reason: ExitReason
    iterations: T.List[IterationData]

class SolverParams:
    solver_iter_max: int
    pcg_iter_max: int

    diag_init: float
    diag_scaling_up: float
    diag_scaling_down: float
    diag_exit_value: float
    diag_min: float

    solver_rel_decrease_min: float
    score_exit_value: float

    pcg_rel_decrease_min: float
    pcg_rel_error_exit: float
    pcg_rel_score_exit: float

class GraphSolver:
    def __init__(self, params: SolverParams,
                 *,
                 PinholeCalib_num_max: int = 0,
                 PinholeFocal_num_max: int = 0,
                 PinholePose_num_max: int = 0,
                 PinholePrincipalPoint_num_max: int = 0,
                 Point_num_max: int = 0,
                 SimpleRadialCalib_num_max: int = 0,
                 SimpleRadialFocalAndDistortion_num_max: int = 0,
                 SimpleRadialPose_num_max: int = 0,
                 SimpleRadialPrincipalPoint_num_max: int = 0,
                 simple_radial_num_max: int = 0,
                 simple_radial_fixed_pose_num_max: int = 0,
                 simple_radial_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_point_num_max: int = 0,
                 pinhole_num_max: int = 0,
                 pinhole_fixed_pose_num_max: int = 0,
                 pinhole_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_point_num_max: int = 0,
                 simple_radial_split_fixed_focal_and_distortion_num_max: int = 0,
                 simple_radial_split_fixed_principal_point_num_max: int = 0,
                 simple_radial_split_fixed_pose_fixed_focal_and_distortion_num_max: int = 0,
                 simple_radial_split_fixed_pose_fixed_principal_point_num_max: int = 0,
                 simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num_max: int = 0,
                 simple_radial_split_fixed_focal_and_distortion_fixed_point_num_max: int = 0,
                 simple_radial_split_fixed_principal_point_fixed_point_num_max: int = 0,
                 simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num_max: int = 0,
                 simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num_max: int = 0,
                 simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max: int = 0,
                 simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_split_fixed_focal_num_max: int = 0,
                 pinhole_split_fixed_principal_point_num_max: int = 0,
                 pinhole_split_fixed_pose_fixed_focal_num_max: int = 0,
                 pinhole_split_fixed_pose_fixed_principal_point_num_max: int = 0,
                 pinhole_split_fixed_focal_fixed_principal_point_num_max: int = 0,
                 pinhole_split_fixed_focal_fixed_point_num_max: int = 0,
                 pinhole_split_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max: int = 0,
                 pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max: int = 0,
                 pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max: int = 0,
    ): ...

    def set_params(self, params: SolverParams) -> None:
        """
        Set the solver parameters.
        """

    def solve(self, print_progress: bool = False, verbose_logging: bool = False) -> SolveResult:
        """
        Run the solver.
        """

    def finish_indices(self) -> None:
        """
        Finish the indices.

        This function has to be called after all indices are set and before the solve function is called.
        """

    def get_allocation_size(self) -> int:
        """
        Get the number of allocated bytes.
        """

    def set_PinholeCalib_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholeCalib nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholeCalib_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholeCalib nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholeCalib_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholeCalib nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholeCalib_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholeCalib nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholeCalib_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholeCalib.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_PinholeFocal_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholeFocal nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholeFocal_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholeFocal nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholeFocal_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholeFocal nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholeFocal_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholeFocal nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholeFocal_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholeFocal.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_PinholePose_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholePose nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholePose_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholePose nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholePose_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholePose nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholePose_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholePose nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholePose_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholePose.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_PinholePrincipalPoint_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholePrincipalPoint nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholePrincipalPoint_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholePrincipalPoint nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholePrincipalPoint_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholePrincipalPoint nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholePrincipalPoint_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholePrincipalPoint nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholePrincipalPoint_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholePrincipalPoint.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_Point_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the Point nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_Point_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the Point nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_Point_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the Point nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_Point_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the Point nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_Point_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type Point.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialCalib_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialCalib nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialCalib_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialCalib nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialCalib_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialCalib nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialCalib_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialCalib nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialCalib_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialCalib.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialFocalAndDistortion_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocalAndDistortion nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialFocalAndDistortion_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocalAndDistortion nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialFocalAndDistortion_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocalAndDistortion nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialFocalAndDistortion_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocalAndDistortion nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialFocalAndDistortion_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialFocalAndDistortion.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialPose_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialPose nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialPose_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialPose nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialPose_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialPose nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialPose_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialPose nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialPose_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialPose.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialPrincipalPoint_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialPrincipalPoint nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialPrincipalPoint_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialPrincipalPoint nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialPrincipalPoint_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialPrincipalPoint nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialPrincipalPoint_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialPrincipalPoint nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialPrincipalPoint_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialPrincipalPoint.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

    def set_simple_radial_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial factor from host.
        """

    def set_simple_radial_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial factor from device.
        """
    def set_simple_radial_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the simple_radial factor from host.
        """

    def set_simple_radial_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the simple_radial factor from device.
        """
    def set_simple_radial_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial factor from host.
        """

    def set_simple_radial_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial factor from device.
        """

    def set_simple_radial_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_num(self, num: int) -> None:
        """
        Set the current number of simple_radial factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_pose factor from device.
        """
    def set_simple_radial_fixed_pose_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose factor from device.
        """

    def set_simple_radial_fixed_pose_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_point factor from device.
        """
    def set_simple_radial_fixed_point_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_point factor from device.
        """

    def set_simple_radial_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_point_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_pose_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_point_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the simple_radial_fixed_pose_fixed_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole factor from host.
        """

    def set_pinhole_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole factor from device.
        """
    def set_pinhole_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the pinhole factor from host.
        """

    def set_pinhole_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the pinhole factor from device.
        """
    def set_pinhole_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole factor from host.
        """

    def set_pinhole_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole factor from device.
        """

    def set_pinhole_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_num(self, num: int) -> None:
        """
        Set the current number of pinhole factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_pose factor from device.
        """
    def set_pinhole_fixed_pose_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose factor from device.
        """

    def set_pinhole_fixed_pose_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_point factor from device.
        """
    def set_pinhole_fixed_point_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_point factor from device.
        """

    def set_pinhole_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_point_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_pose_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_point_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the calib argument for the pinhole_fixed_pose_fixed_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion factor from device.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_focal_and_distortion factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_focal_and_distortion factor from device.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_focal_and_distortion factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_focal_and_distortion factor from device.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_focal_and_distortion factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_principal_point factor from device.
        """
    def set_simple_radial_split_fixed_principal_point_focal_and_distortion_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_principal_point_focal_and_distortion_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_principal_point factor from device.
        """
    def set_simple_radial_split_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_principal_point factor from device.
        """

    def set_simple_radial_split_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from device.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from device.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_pose_fixed_focal_and_distortion factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_focal_and_distortion_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_pose_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_focal_and_distortion_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_pose_fixed_principal_point factor from device.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_principal_point factor from device.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_pose_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from device.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from device.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_focal_and_distortion_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_point factor from device.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_focal_and_distortion_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_focal_and_distortion_fixed_point factor from device.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_focal_and_distortion_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_principal_point_fixed_point factor from device.
        """
    def set_simple_radial_split_fixed_principal_point_fixed_point_focal_and_distortion_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_focal_and_distortion_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from device.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from device.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_focal_and_distortion_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_focal_and_distortion_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_distortion argument for the simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_pose_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_focal_and_distortion_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_focal_and_distortion_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_distortion consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_focal_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal factor from host.
        """

    def set_pinhole_split_fixed_focal_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal factor from device.
        """
    def set_pinhole_split_fixed_focal_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_focal factor from host.
        """

    def set_pinhole_split_fixed_focal_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_focal factor from device.
        """
    def set_pinhole_split_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_focal factor from host.
        """

    def set_pinhole_split_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_focal factor from device.
        """

    def set_pinhole_split_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_principal_point factor from device.
        """
    def set_pinhole_split_fixed_principal_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_principal_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_principal_point factor from device.
        """
    def set_pinhole_split_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_principal_point factor from device.
        """

    def set_pinhole_split_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_pose_fixed_focal factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_pose_fixed_focal factor from device.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_focal factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_focal factor from device.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_pose_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_pose_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_pose_fixed_principal_point factor from device.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_principal_point factor from device.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_pose_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_principal_point factor from device.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_focal_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_focal_fixed_principal_point factor from device.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_focal_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_focal_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_focal_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_point factor from device.
        """
    def set_pinhole_split_fixed_focal_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_focal_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_focal_fixed_point factor from device.
        """

    def set_pinhole_split_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_principal_point_fixed_point factor from device.
        """
    def set_pinhole_split_fixed_principal_point_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from device.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_pose_fixed_focal_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_pose_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_split_fixed_pose_fixed_focal_fixed_point factor from device.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_pose_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_pose_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_split_fixed_focal_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_split_fixed_focal_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

def const_pinhole_focal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholeFocal data to the caspar data format.
    """

def const_pinhole_focal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholeFocal data to the stacked data format.
    """

def const_pinhole_pose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholePose data to the caspar data format.
    """

def const_pinhole_pose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholePose data to the stacked data format.
    """

def const_pinhole_principal_point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholePrincipalPoint data to the caspar data format.
    """

def const_pinhole_principal_point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholePrincipalPoint data to the stacked data format.
    """

def const_pixel_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPixel data to the caspar data format.
    """

def const_pixel_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPixel data to the stacked data format.
    """

def const_point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPoint data to the caspar data format.
    """

def const_point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPoint data to the stacked data format.
    """

def const_simple_radial_focal_and_distortion_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialFocalAndDistortion data to the caspar data format.
    """

def const_simple_radial_focal_and_distortion_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialFocalAndDistortion data to the stacked data format.
    """

def const_simple_radial_pose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialPose data to the caspar data format.
    """

def const_simple_radial_pose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialPose data to the stacked data format.
    """

def const_simple_radial_principal_point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialPrincipalPoint data to the caspar data format.
    """

def const_simple_radial_principal_point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialPrincipalPoint data to the stacked data format.
    """

def pinhole_calib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeCalib data to the caspar data format.
    """

def pinhole_calib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeCalib data to the stacked data format.
    """

def pinhole_focal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeFocal data to the caspar data format.
    """

def pinhole_focal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeFocal data to the stacked data format.
    """

def pinhole_pose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholePose data to the caspar data format.
    """

def pinhole_pose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholePose data to the stacked data format.
    """

def pinhole_principal_point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholePrincipalPoint data to the caspar data format.
    """

def pinhole_principal_point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholePrincipalPoint data to the stacked data format.
    """

def point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked Point data to the caspar data format.
    """

def point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar Point data to the stacked data format.
    """

def simple_radial_calib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialCalib data to the caspar data format.
    """

def simple_radial_calib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialCalib data to the stacked data format.
    """

def simple_radial_focal_and_distortion_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialFocalAndDistortion data to the caspar data format.
    """

def simple_radial_focal_and_distortion_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialFocalAndDistortion data to the stacked data format.
    """

def simple_radial_pose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialPose data to the caspar data format.
    """

def simple_radial_pose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialPose data to the stacked data format.
    """

def simple_radial_principal_point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialPrincipalPoint data to the caspar data format.
    """

def simple_radial_principal_point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialPrincipalPoint data to the stacked data format.
    """


def shared_indices(indices: CudaArray, out_shared: CudaArray) -> None:
    """
    Calculate shared indices from the indices.
    """

def simple_radial_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_res_jac_first(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_res_jac(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_score(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_jtjnjtr_direct(
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_res_jac_first(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_res_jac(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_score(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_res_jac_first(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_res_jac(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_score(
    calib: CudaArray,
    calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_jtjnjtr_direct(
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_res_jac_first(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_res_jac(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_calib_njtr: CudaArray,
    out_calib_precond_diag: CudaArray,
    out_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_score(
    calib: CudaArray,
    calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_jtjnjtr_direct(
    calib_njtr: CudaArray,
    calib_njtr_indices: CudaArray,
    calib_jac: CudaArray,
    out_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_distortion_njtr: CudaArray,
    focal_and_distortion_njtr_indices: CudaArray,
    focal_and_distortion_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    out_res: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_res_jac_first(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_res_jac(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_score(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_jtjnjtr_direct(
    focal_and_distortion_njtr: CudaArray,
    focal_and_distortion_njtr_indices: CudaArray,
    focal_and_distortion_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_distortion_njtr: CudaArray,
    focal_and_distortion_njtr_indices: CudaArray,
    focal_and_distortion_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_distortion: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_res_jac(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    out_focal_and_distortion_precond_diag: CudaArray,
    out_focal_and_distortion_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_score(
    focal_and_distortion: CudaArray,
    focal_and_distortion_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct(
    focal_and_distortion_njtr: CudaArray,
    focal_and_distortion_njtr_indices: CudaArray,
    focal_and_distortion_jac: CudaArray,
    out_focal_and_distortion_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_distortion: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_point_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_fixed_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_split_fixed_focal_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_retract(
    PinholeCalib: CudaArray,
    delta: CudaArray,
    out_PinholeCalib_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_start_w(
    PinholeCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeCalib_p: CudaArray,
    out_PinholeCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_start_w_contribute(
    PinholeCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeCalib_p: CudaArray,
    out_PinholeCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_alpha_numerator_denominator(
    PinholeCalib_p_kp1: CudaArray,
    PinholeCalib_r_k: CudaArray,
    PinholeCalib_w: CudaArray,
    PinholeCalib_total_ag: CudaArray,
    PinholeCalib_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_alpha_denominator_or_beta_numerator(
    PinholeCalib_p_kp1: CudaArray,
    PinholeCalib_w: CudaArray,
    PinholeCalib_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_r_first(
    PinholeCalib_r_k: CudaArray,
    PinholeCalib_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeCalib_r_kp1: CudaArray,
    out_PinholeCalib_r_0_norm2_tot: CudaArray,
    out_PinholeCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_r(
    PinholeCalib_r_k: CudaArray,
    PinholeCalib_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeCalib_r_kp1: CudaArray,
    out_PinholeCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_step_first(
    PinholeCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_step(
    PinholeCalib_step_k: CudaArray,
    PinholeCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_p(
    PinholeCalib_z: CudaArray,
    PinholeCalib_p_k: CudaArray,
    beta: CudaArray,
    out_PinholeCalib_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_update_Mp(
    PinholeCalib_r_k: CudaArray,
    PinholeCalib_Mp: CudaArray,
    beta: CudaArray,
    out_PinholeCalib_Mp_kp1: CudaArray,
    out_PinholeCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeCalib_pred_decrease_times_two(
    PinholeCalib_step: CudaArray,
    PinholeCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeCalib_njtr: CudaArray,
    out_PinholeCalib_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_retract(
    PinholeFocal: CudaArray,
    delta: CudaArray,
    out_PinholeFocal_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_start_w(
    PinholeFocal_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocal_p: CudaArray,
    out_PinholeFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_start_w_contribute(
    PinholeFocal_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocal_p: CudaArray,
    out_PinholeFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_alpha_numerator_denominator(
    PinholeFocal_p_kp1: CudaArray,
    PinholeFocal_r_k: CudaArray,
    PinholeFocal_w: CudaArray,
    PinholeFocal_total_ag: CudaArray,
    PinholeFocal_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_alpha_denominator_or_beta_numerator(
    PinholeFocal_p_kp1: CudaArray,
    PinholeFocal_w: CudaArray,
    PinholeFocal_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_r_first(
    PinholeFocal_r_k: CudaArray,
    PinholeFocal_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeFocal_r_kp1: CudaArray,
    out_PinholeFocal_r_0_norm2_tot: CudaArray,
    out_PinholeFocal_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_r(
    PinholeFocal_r_k: CudaArray,
    PinholeFocal_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeFocal_r_kp1: CudaArray,
    out_PinholeFocal_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_step_first(
    PinholeFocal_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeFocal_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_step(
    PinholeFocal_step_k: CudaArray,
    PinholeFocal_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeFocal_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_p(
    PinholeFocal_z: CudaArray,
    PinholeFocal_p_k: CudaArray,
    beta: CudaArray,
    out_PinholeFocal_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_update_Mp(
    PinholeFocal_r_k: CudaArray,
    PinholeFocal_Mp: CudaArray,
    beta: CudaArray,
    out_PinholeFocal_Mp_kp1: CudaArray,
    out_PinholeFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocal_pred_decrease_times_two(
    PinholeFocal_step: CudaArray,
    PinholeFocal_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocal_njtr: CudaArray,
    out_PinholeFocal_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_retract(
    PinholePose: CudaArray,
    delta: CudaArray,
    out_PinholePose_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_start_w(
    PinholePose_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePose_p: CudaArray,
    out_PinholePose_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_start_w_contribute(
    PinholePose_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePose_p: CudaArray,
    out_PinholePose_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_alpha_numerator_denominator(
    PinholePose_p_kp1: CudaArray,
    PinholePose_r_k: CudaArray,
    PinholePose_w: CudaArray,
    PinholePose_total_ag: CudaArray,
    PinholePose_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_alpha_denominator_or_beta_numerator(
    PinholePose_p_kp1: CudaArray,
    PinholePose_w: CudaArray,
    PinholePose_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_r_first(
    PinholePose_r_k: CudaArray,
    PinholePose_w: CudaArray,
    negalpha: CudaArray,
    out_PinholePose_r_kp1: CudaArray,
    out_PinholePose_r_0_norm2_tot: CudaArray,
    out_PinholePose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_r(
    PinholePose_r_k: CudaArray,
    PinholePose_w: CudaArray,
    negalpha: CudaArray,
    out_PinholePose_r_kp1: CudaArray,
    out_PinholePose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_step_first(
    PinholePose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholePose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_step(
    PinholePose_step_k: CudaArray,
    PinholePose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholePose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_p(
    PinholePose_z: CudaArray,
    PinholePose_p_k: CudaArray,
    beta: CudaArray,
    out_PinholePose_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_update_Mp(
    PinholePose_r_k: CudaArray,
    PinholePose_Mp: CudaArray,
    beta: CudaArray,
    out_PinholePose_Mp_kp1: CudaArray,
    out_PinholePose_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePose_pred_decrease_times_two(
    PinholePose_step: CudaArray,
    PinholePose_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePose_njtr: CudaArray,
    out_PinholePose_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_retract(
    PinholePrincipalPoint: CudaArray,
    delta: CudaArray,
    out_PinholePrincipalPoint_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_start_w(
    PinholePrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePrincipalPoint_p: CudaArray,
    out_PinholePrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_start_w_contribute(
    PinholePrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePrincipalPoint_p: CudaArray,
    out_PinholePrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_alpha_numerator_denominator(
    PinholePrincipalPoint_p_kp1: CudaArray,
    PinholePrincipalPoint_r_k: CudaArray,
    PinholePrincipalPoint_w: CudaArray,
    PinholePrincipalPoint_total_ag: CudaArray,
    PinholePrincipalPoint_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_alpha_denominator_or_beta_numerator(
    PinholePrincipalPoint_p_kp1: CudaArray,
    PinholePrincipalPoint_w: CudaArray,
    PinholePrincipalPoint_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_r_first(
    PinholePrincipalPoint_r_k: CudaArray,
    PinholePrincipalPoint_w: CudaArray,
    negalpha: CudaArray,
    out_PinholePrincipalPoint_r_kp1: CudaArray,
    out_PinholePrincipalPoint_r_0_norm2_tot: CudaArray,
    out_PinholePrincipalPoint_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_r(
    PinholePrincipalPoint_r_k: CudaArray,
    PinholePrincipalPoint_w: CudaArray,
    negalpha: CudaArray,
    out_PinholePrincipalPoint_r_kp1: CudaArray,
    out_PinholePrincipalPoint_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_step_first(
    PinholePrincipalPoint_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholePrincipalPoint_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_step(
    PinholePrincipalPoint_step_k: CudaArray,
    PinholePrincipalPoint_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholePrincipalPoint_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_p(
    PinholePrincipalPoint_z: CudaArray,
    PinholePrincipalPoint_p_k: CudaArray,
    beta: CudaArray,
    out_PinholePrincipalPoint_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_update_Mp(
    PinholePrincipalPoint_r_k: CudaArray,
    PinholePrincipalPoint_Mp: CudaArray,
    beta: CudaArray,
    out_PinholePrincipalPoint_Mp_kp1: CudaArray,
    out_PinholePrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholePrincipalPoint_pred_decrease_times_two(
    PinholePrincipalPoint_step: CudaArray,
    PinholePrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    PinholePrincipalPoint_njtr: CudaArray,
    out_PinholePrincipalPoint_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def Point_retract(
    Point: CudaArray,
    delta: CudaArray,
    out_Point_retracted: CudaArray,
    problem_size: int
) -> None: ...

def Point_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def Point_start_w(
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_p: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_start_w_contribute(
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_p: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_alpha_numerator_denominator(
    Point_p_kp1: CudaArray,
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    Point_total_ag: CudaArray,
    Point_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def Point_alpha_denominator_or_beta_numerator(
    Point_p_kp1: CudaArray,
    Point_w: CudaArray,
    Point_out: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_r_first(
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    negalpha: CudaArray,
    out_Point_r_kp1: CudaArray,
    out_Point_r_0_norm2_tot: CudaArray,
    out_Point_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_r(
    Point_r_k: CudaArray,
    Point_w: CudaArray,
    negalpha: CudaArray,
    out_Point_r_kp1: CudaArray,
    out_Point_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_step_first(
    Point_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Point_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_step(
    Point_step_k: CudaArray,
    Point_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Point_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_p(
    Point_z: CudaArray,
    Point_p_k: CudaArray,
    beta: CudaArray,
    out_Point_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Point_update_Mp(
    Point_r_k: CudaArray,
    Point_Mp: CudaArray,
    beta: CudaArray,
    out_Point_Mp_kp1: CudaArray,
    out_Point_w: CudaArray,
    problem_size: int
) -> None: ...

def Point_pred_decrease_times_two(
    Point_step: CudaArray,
    Point_precond_diag: CudaArray,
    diag: CudaArray,
    Point_njtr: CudaArray,
    out_Point_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_retract(
    SimpleRadialCalib: CudaArray,
    delta: CudaArray,
    out_SimpleRadialCalib_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_start_w(
    SimpleRadialCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialCalib_p: CudaArray,
    out_SimpleRadialCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_start_w_contribute(
    SimpleRadialCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialCalib_p: CudaArray,
    out_SimpleRadialCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_alpha_numerator_denominator(
    SimpleRadialCalib_p_kp1: CudaArray,
    SimpleRadialCalib_r_k: CudaArray,
    SimpleRadialCalib_w: CudaArray,
    SimpleRadialCalib_total_ag: CudaArray,
    SimpleRadialCalib_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_alpha_denominator_or_beta_numerator(
    SimpleRadialCalib_p_kp1: CudaArray,
    SimpleRadialCalib_w: CudaArray,
    SimpleRadialCalib_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_r_first(
    SimpleRadialCalib_r_k: CudaArray,
    SimpleRadialCalib_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialCalib_r_kp1: CudaArray,
    out_SimpleRadialCalib_r_0_norm2_tot: CudaArray,
    out_SimpleRadialCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_r(
    SimpleRadialCalib_r_k: CudaArray,
    SimpleRadialCalib_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialCalib_r_kp1: CudaArray,
    out_SimpleRadialCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_step_first(
    SimpleRadialCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_step(
    SimpleRadialCalib_step_k: CudaArray,
    SimpleRadialCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_p(
    SimpleRadialCalib_z: CudaArray,
    SimpleRadialCalib_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialCalib_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_update_Mp(
    SimpleRadialCalib_r_k: CudaArray,
    SimpleRadialCalib_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialCalib_Mp_kp1: CudaArray,
    out_SimpleRadialCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialCalib_pred_decrease_times_two(
    SimpleRadialCalib_step: CudaArray,
    SimpleRadialCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialCalib_njtr: CudaArray,
    out_SimpleRadialCalib_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_retract(
    SimpleRadialFocalAndDistortion: CudaArray,
    delta: CudaArray,
    out_SimpleRadialFocalAndDistortion_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_start_w(
    SimpleRadialFocalAndDistortion_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndDistortion_p: CudaArray,
    out_SimpleRadialFocalAndDistortion_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_start_w_contribute(
    SimpleRadialFocalAndDistortion_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndDistortion_p: CudaArray,
    out_SimpleRadialFocalAndDistortion_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_alpha_numerator_denominator(
    SimpleRadialFocalAndDistortion_p_kp1: CudaArray,
    SimpleRadialFocalAndDistortion_r_k: CudaArray,
    SimpleRadialFocalAndDistortion_w: CudaArray,
    SimpleRadialFocalAndDistortion_total_ag: CudaArray,
    SimpleRadialFocalAndDistortion_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_alpha_denominator_or_beta_numerator(
    SimpleRadialFocalAndDistortion_p_kp1: CudaArray,
    SimpleRadialFocalAndDistortion_w: CudaArray,
    SimpleRadialFocalAndDistortion_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_r_first(
    SimpleRadialFocalAndDistortion_r_k: CudaArray,
    SimpleRadialFocalAndDistortion_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocalAndDistortion_r_kp1: CudaArray,
    out_SimpleRadialFocalAndDistortion_r_0_norm2_tot: CudaArray,
    out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_r(
    SimpleRadialFocalAndDistortion_r_k: CudaArray,
    SimpleRadialFocalAndDistortion_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocalAndDistortion_r_kp1: CudaArray,
    out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_step_first(
    SimpleRadialFocalAndDistortion_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocalAndDistortion_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_step(
    SimpleRadialFocalAndDistortion_step_k: CudaArray,
    SimpleRadialFocalAndDistortion_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocalAndDistortion_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_p(
    SimpleRadialFocalAndDistortion_z: CudaArray,
    SimpleRadialFocalAndDistortion_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocalAndDistortion_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_update_Mp(
    SimpleRadialFocalAndDistortion_r_k: CudaArray,
    SimpleRadialFocalAndDistortion_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocalAndDistortion_Mp_kp1: CudaArray,
    out_SimpleRadialFocalAndDistortion_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndDistortion_pred_decrease_times_two(
    SimpleRadialFocalAndDistortion_step: CudaArray,
    SimpleRadialFocalAndDistortion_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndDistortion_njtr: CudaArray,
    out_SimpleRadialFocalAndDistortion_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_retract(
    SimpleRadialPose: CudaArray,
    delta: CudaArray,
    out_SimpleRadialPose_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_start_w(
    SimpleRadialPose_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPose_p: CudaArray,
    out_SimpleRadialPose_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_start_w_contribute(
    SimpleRadialPose_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPose_p: CudaArray,
    out_SimpleRadialPose_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_alpha_numerator_denominator(
    SimpleRadialPose_p_kp1: CudaArray,
    SimpleRadialPose_r_k: CudaArray,
    SimpleRadialPose_w: CudaArray,
    SimpleRadialPose_total_ag: CudaArray,
    SimpleRadialPose_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_alpha_denominator_or_beta_numerator(
    SimpleRadialPose_p_kp1: CudaArray,
    SimpleRadialPose_w: CudaArray,
    SimpleRadialPose_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_r_first(
    SimpleRadialPose_r_k: CudaArray,
    SimpleRadialPose_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialPose_r_kp1: CudaArray,
    out_SimpleRadialPose_r_0_norm2_tot: CudaArray,
    out_SimpleRadialPose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_r(
    SimpleRadialPose_r_k: CudaArray,
    SimpleRadialPose_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialPose_r_kp1: CudaArray,
    out_SimpleRadialPose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_step_first(
    SimpleRadialPose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialPose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_step(
    SimpleRadialPose_step_k: CudaArray,
    SimpleRadialPose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialPose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_p(
    SimpleRadialPose_z: CudaArray,
    SimpleRadialPose_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialPose_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_update_Mp(
    SimpleRadialPose_r_k: CudaArray,
    SimpleRadialPose_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialPose_Mp_kp1: CudaArray,
    out_SimpleRadialPose_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPose_pred_decrease_times_two(
    SimpleRadialPose_step: CudaArray,
    SimpleRadialPose_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPose_njtr: CudaArray,
    out_SimpleRadialPose_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_retract(
    SimpleRadialPrincipalPoint: CudaArray,
    delta: CudaArray,
    out_SimpleRadialPrincipalPoint_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_start_w(
    SimpleRadialPrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPrincipalPoint_p: CudaArray,
    out_SimpleRadialPrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_start_w_contribute(
    SimpleRadialPrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPrincipalPoint_p: CudaArray,
    out_SimpleRadialPrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_alpha_numerator_denominator(
    SimpleRadialPrincipalPoint_p_kp1: CudaArray,
    SimpleRadialPrincipalPoint_r_k: CudaArray,
    SimpleRadialPrincipalPoint_w: CudaArray,
    SimpleRadialPrincipalPoint_total_ag: CudaArray,
    SimpleRadialPrincipalPoint_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_alpha_denominator_or_beta_numerator(
    SimpleRadialPrincipalPoint_p_kp1: CudaArray,
    SimpleRadialPrincipalPoint_w: CudaArray,
    SimpleRadialPrincipalPoint_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_r_first(
    SimpleRadialPrincipalPoint_r_k: CudaArray,
    SimpleRadialPrincipalPoint_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialPrincipalPoint_r_kp1: CudaArray,
    out_SimpleRadialPrincipalPoint_r_0_norm2_tot: CudaArray,
    out_SimpleRadialPrincipalPoint_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_r(
    SimpleRadialPrincipalPoint_r_k: CudaArray,
    SimpleRadialPrincipalPoint_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialPrincipalPoint_r_kp1: CudaArray,
    out_SimpleRadialPrincipalPoint_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_step_first(
    SimpleRadialPrincipalPoint_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialPrincipalPoint_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_step(
    SimpleRadialPrincipalPoint_step_k: CudaArray,
    SimpleRadialPrincipalPoint_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialPrincipalPoint_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_p(
    SimpleRadialPrincipalPoint_z: CudaArray,
    SimpleRadialPrincipalPoint_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialPrincipalPoint_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_update_Mp(
    SimpleRadialPrincipalPoint_r_k: CudaArray,
    SimpleRadialPrincipalPoint_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialPrincipalPoint_Mp_kp1: CudaArray,
    out_SimpleRadialPrincipalPoint_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialPrincipalPoint_pred_decrease_times_two(
    SimpleRadialPrincipalPoint_step: CudaArray,
    SimpleRadialPrincipalPoint_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialPrincipalPoint_njtr: CudaArray,
    out_SimpleRadialPrincipalPoint_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

