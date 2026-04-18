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
                 PinholeExtraCalib_num_max: int = 0,
                 PinholeFocal_num_max: int = 0,
                 Point_num_max: int = 0,
                 Pose_num_max: int = 0,
                 SimpleRadialExtraCalib_num_max: int = 0,
                 SimpleRadialFocal_num_max: int = 0,
                 simple_radial_num_max: int = 0,
                 simple_radial_fixed_pose_num_max: int = 0,
                 simple_radial_fixed_focal_num_max: int = 0,
                 simple_radial_fixed_extra_calib_num_max: int = 0,
                 simple_radial_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_extra_calib_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_point_num_max: int = 0,
                 simple_radial_fixed_focal_fixed_extra_calib_num_max: int = 0,
                 simple_radial_fixed_focal_fixed_point_num_max: int = 0,
                 simple_radial_fixed_extra_calib_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num_max: int = 0,
                 simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num_max: int = 0,
                 pinhole_num_max: int = 0,
                 pinhole_fixed_pose_num_max: int = 0,
                 pinhole_fixed_focal_num_max: int = 0,
                 pinhole_fixed_extra_calib_num_max: int = 0,
                 pinhole_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_num_max: int = 0,
                 pinhole_fixed_pose_fixed_extra_calib_num_max: int = 0,
                 pinhole_fixed_pose_fixed_point_num_max: int = 0,
                 pinhole_fixed_focal_fixed_extra_calib_num_max: int = 0,
                 pinhole_fixed_focal_fixed_point_num_max: int = 0,
                 pinhole_fixed_extra_calib_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_extra_calib_fixed_point_num_max: int = 0,
                 pinhole_fixed_focal_fixed_extra_calib_fixed_point_num_max: int = 0,
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

    def set_PinholeExtraCalib_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholeExtraCalib nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholeExtraCalib_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholeExtraCalib nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholeExtraCalib_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholeExtraCalib nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholeExtraCalib_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholeExtraCalib nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholeExtraCalib_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholeExtraCalib.

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
    def set_Pose_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the Pose nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_Pose_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the Pose nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_Pose_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the Pose nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_Pose_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the Pose nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_Pose_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type Pose.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialExtraCalib_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialExtraCalib nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialExtraCalib_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialExtraCalib nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialExtraCalib_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialExtraCalib nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialExtraCalib_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialExtraCalib nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialExtraCalib_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialExtraCalib.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_SimpleRadialFocal_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocal nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialFocal_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocal nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialFocal_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocal nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialFocal_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocal nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialFocal_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialFocal.

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
    def set_simple_radial_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial factor from host.
        """

    def set_simple_radial_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial factor from device.
        """
    def set_simple_radial_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial factor from host.
        """

    def set_simple_radial_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial factor from device.
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
    def set_simple_radial_fixed_pose_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose factor from device.
        """
    def set_simple_radial_fixed_pose_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose factor from device.
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
    def set_simple_radial_fixed_focal_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal factor from host.
        """

    def set_simple_radial_fixed_focal_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal factor from device.
        """
    def set_simple_radial_fixed_focal_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_focal factor from host.
        """

    def set_simple_radial_fixed_focal_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_focal factor from device.
        """
    def set_simple_radial_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal factor from host.
        """

    def set_simple_radial_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal factor from device.
        """

    def set_simple_radial_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_extra_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_extra_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_extra_calib factor from device.
        """
    def set_simple_radial_fixed_extra_calib_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_extra_calib_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_extra_calib factor from device.
        """
    def set_simple_radial_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_extra_calib factor from device.
        """

    def set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_extra_calib factors.

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
    def set_simple_radial_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_point factor from device.
        """
    def set_simple_radial_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_point factor from device.
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
    def set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_focal factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_focal factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_extra_calib factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_extra_calib factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_point factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_point factor from device.
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
    def set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_extra_calib factor from device.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_fixed_extra_calib factor from device.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_focal_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_point factor from device.
        """
    def set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_focal_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_focal_fixed_point factor from device.
        """

    def set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_extra_calib_fixed_point factor from host.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_extra_calib_fixed_point factor from device.
        """
    def set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_extra_calib_fixed_point factor from host.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_extra_calib_fixed_point factor from device.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_extra_calib_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_focal_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the simple_radial_fixed_pose_fixed_focal_fixed_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_extra_calib_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from device.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_fixed_extra_calib_fixed_point factors.

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
    def set_pinhole_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole factor from host.
        """

    def set_pinhole_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole factor from device.
        """
    def set_pinhole_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole factor from host.
        """

    def set_pinhole_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole factor from device.
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
    def set_pinhole_fixed_pose_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose factor from device.
        """
    def set_pinhole_fixed_pose_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose factor from device.
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
    def set_pinhole_fixed_focal_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal factor from host.
        """

    def set_pinhole_fixed_focal_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal factor from device.
        """
    def set_pinhole_fixed_focal_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_focal factor from host.
        """

    def set_pinhole_fixed_focal_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_focal factor from device.
        """
    def set_pinhole_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal factor from host.
        """

    def set_pinhole_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal factor from device.
        """

    def set_pinhole_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_extra_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_extra_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_extra_calib factor from device.
        """
    def set_pinhole_fixed_extra_calib_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_extra_calib_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_extra_calib factor from device.
        """
    def set_pinhole_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_extra_calib factor from device.
        """

    def set_pinhole_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_extra_calib factors.

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
    def set_pinhole_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_point factor from device.
        """
    def set_pinhole_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_point factor from device.
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
    def set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_focal factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_focal factor from device.
        """
    def set_pinhole_fixed_pose_fixed_focal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_extra_calib factor from device.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_extra_calib factor from device.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_point factor from device.
        """
    def set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_point factor from device.
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
    def set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_extra_calib factor from device.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_fixed_extra_calib factor from device.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_focal_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_point factor from device.
        """
    def set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_focal_fixed_point factor from device.
        """

    def set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_extra_calib_fixed_point factor from host.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_extra_calib_fixed_point factor from device.
        """
    def set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_extra_calib_fixed_point factor from host.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_extra_calib_fixed_point factor from device.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_extra_calib_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_focal_fixed_extra_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_extra_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal_fixed_extra_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_focal_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_extra_calib_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the extra_calib argument for the pinhole_fixed_pose_fixed_focal_fixed_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_focal_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_focal_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal argument for the pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_extra_calib_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from device.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_focal_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_extra_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the extra_calib consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_fixed_extra_calib_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_fixed_extra_calib_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_fixed_extra_calib_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

def ConstPinholeExtraCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholeExtraCalib data to the caspar data format.
    """

def ConstPinholeExtraCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholeExtraCalib data to the stacked data format.
    """

def ConstPinholeFocal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholeFocal data to the caspar data format.
    """

def ConstPinholeFocal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholeFocal data to the stacked data format.
    """

def ConstPixel_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPixel data to the caspar data format.
    """

def ConstPixel_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPixel data to the stacked data format.
    """

def ConstPoint_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPoint data to the caspar data format.
    """

def ConstPoint_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPoint data to the stacked data format.
    """

def ConstPose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPose data to the caspar data format.
    """

def ConstPose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPose data to the stacked data format.
    """

def ConstSimpleRadialExtraCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialExtraCalib data to the caspar data format.
    """

def ConstSimpleRadialExtraCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialExtraCalib data to the stacked data format.
    """

def ConstSimpleRadialFocal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialFocal data to the caspar data format.
    """

def ConstSimpleRadialFocal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialFocal data to the stacked data format.
    """

def PinholeExtraCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeExtraCalib data to the caspar data format.
    """

def PinholeExtraCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeExtraCalib data to the stacked data format.
    """

def PinholeFocal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeFocal data to the caspar data format.
    """

def PinholeFocal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeFocal data to the stacked data format.
    """

def Point_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked Point data to the caspar data format.
    """

def Point_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar Point data to the stacked data format.
    """

def Pose_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked Pose data to the caspar data format.
    """

def Pose_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar Pose data to the stacked data format.
    """

def SimpleRadialExtraCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialExtraCalib data to the caspar data format.
    """

def SimpleRadialExtraCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialExtraCalib data to the stacked data format.
    """

def SimpleRadialFocal_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialFocal data to the caspar data format.
    """

def SimpleRadialFocal_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialFocal data to the stacked data format.
    """


def shared_indices(indices: CudaArray, out_shared: CudaArray) -> None:
    """
    Calculate shared indices from the indices.
    """

def simple_radial_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
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
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_extra_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_extra_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_extra_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_extra_calib_jtjnjtr_direct(
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

def simple_radial_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_res_jac_first(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_res_jac(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_score(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_jtjnjtr_direct(
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_pose_fixed_extra_calib_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_pose_fixed_extra_calib_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_jtjnjtr_direct(
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

def simple_radial_fixed_pose_fixed_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_focal_fixed_extra_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_focal_fixed_extra_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
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

def simple_radial_fixed_focal_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_extra_calib_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_extra_calib_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def simple_radial_fixed_extra_calib_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_extra_calib_fixed_point_jtjnjtr_direct(
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

def simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac_first(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_point_res_jac(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_point_score(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct(
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_fixed_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_extra_calib_fixed_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_fixed_extra_calib_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
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
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_extra_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_extra_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_extra_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_extra_calib_jtjnjtr_direct(
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

def pinhole_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
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
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_res_jac_first(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_res_jac(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_res: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_score(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_jtjnjtr_direct(
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_pose_fixed_extra_calib_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_pose_fixed_extra_calib_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_jtjnjtr_direct(
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

def pinhole_fixed_pose_fixed_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_focal_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_focal_fixed_extra_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_focal_fixed_extra_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
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

def pinhole_fixed_focal_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_extra_calib_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_extra_calib_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
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

def pinhole_fixed_extra_calib_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_extra_calib_fixed_point_jtjnjtr_direct(
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

def pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_extra_calib_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_point_res_jac_first(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_point_res_jac(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_extra_calib_njtr: CudaArray,
    out_extra_calib_precond_diag: CudaArray,
    out_extra_calib_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_point_score(
    extra_calib: CudaArray,
    extra_calib_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_fixed_point_jtjnjtr_direct(
    extra_calib_njtr: CudaArray,
    extra_calib_njtr_indices: CudaArray,
    extra_calib_jac: CudaArray,
    out_extra_calib_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac_first(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_njtr: CudaArray,
    out_focal_precond_diag: CudaArray,
    out_focal_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_fixed_point_score(
    focal: CudaArray,
    focal_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_extra_calib_fixed_point_jtjnjtr_direct(
    focal_njtr: CudaArray,
    focal_njtr_indices: CudaArray,
    focal_jac: CudaArray,
    out_focal_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal: CudaArray,
    extra_calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_fixed_extra_calib_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_retract(
    PinholeExtraCalib: CudaArray,
    delta: CudaArray,
    out_PinholeExtraCalib_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_start_w(
    PinholeExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeExtraCalib_p: CudaArray,
    out_PinholeExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_start_w_contribute(
    PinholeExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeExtraCalib_p: CudaArray,
    out_PinholeExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_alpha_numerator_denominator(
    PinholeExtraCalib_p_kp1: CudaArray,
    PinholeExtraCalib_r_k: CudaArray,
    PinholeExtraCalib_w: CudaArray,
    PinholeExtraCalib_total_ag: CudaArray,
    PinholeExtraCalib_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_alpha_denumerator_or_beta_nummerator(
    PinholeExtraCalib_p_kp1: CudaArray,
    PinholeExtraCalib_w: CudaArray,
    PinholeExtraCalib_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_r_first(
    PinholeExtraCalib_r_k: CudaArray,
    PinholeExtraCalib_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeExtraCalib_r_kp1: CudaArray,
    out_PinholeExtraCalib_r_0_norm2_tot: CudaArray,
    out_PinholeExtraCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_r(
    PinholeExtraCalib_r_k: CudaArray,
    PinholeExtraCalib_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeExtraCalib_r_kp1: CudaArray,
    out_PinholeExtraCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_step_first(
    PinholeExtraCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeExtraCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_step(
    PinholeExtraCalib_step_k: CudaArray,
    PinholeExtraCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeExtraCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_p(
    PinholeExtraCalib_z: CudaArray,
    PinholeExtraCalib_p_k: CudaArray,
    beta: CudaArray,
    out_PinholeExtraCalib_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_update_Mp(
    PinholeExtraCalib_r_k: CudaArray,
    PinholeExtraCalib_Mp: CudaArray,
    beta: CudaArray,
    out_PinholeExtraCalib_Mp_kp1: CudaArray,
    out_PinholeExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeExtraCalib_pred_decrease_times_two(
    PinholeExtraCalib_step: CudaArray,
    PinholeExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeExtraCalib_njtr: CudaArray,
    out_PinholeExtraCalib_pred_dec: CudaArray,
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

def PinholeFocal_alpha_denumerator_or_beta_nummerator(
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

def Point_alpha_denumerator_or_beta_nummerator(
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

def Pose_retract(
    Pose: CudaArray,
    delta: CudaArray,
    out_Pose_retracted: CudaArray,
    problem_size: int
) -> None: ...

def Pose_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def Pose_start_w(
    Pose_precond_diag: CudaArray,
    diag: CudaArray,
    Pose_p: CudaArray,
    out_Pose_w: CudaArray,
    problem_size: int
) -> None: ...

def Pose_start_w_contribute(
    Pose_precond_diag: CudaArray,
    diag: CudaArray,
    Pose_p: CudaArray,
    out_Pose_w: CudaArray,
    problem_size: int
) -> None: ...

def Pose_alpha_numerator_denominator(
    Pose_p_kp1: CudaArray,
    Pose_r_k: CudaArray,
    Pose_w: CudaArray,
    Pose_total_ag: CudaArray,
    Pose_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def Pose_alpha_denumerator_or_beta_nummerator(
    Pose_p_kp1: CudaArray,
    Pose_w: CudaArray,
    Pose_out: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_r_first(
    Pose_r_k: CudaArray,
    Pose_w: CudaArray,
    negalpha: CudaArray,
    out_Pose_r_kp1: CudaArray,
    out_Pose_r_0_norm2_tot: CudaArray,
    out_Pose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_r(
    Pose_r_k: CudaArray,
    Pose_w: CudaArray,
    negalpha: CudaArray,
    out_Pose_r_kp1: CudaArray,
    out_Pose_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_step_first(
    Pose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Pose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_step(
    Pose_step_k: CudaArray,
    Pose_p_kp1: CudaArray,
    alpha: CudaArray,
    out_Pose_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_p(
    Pose_z: CudaArray,
    Pose_p_k: CudaArray,
    beta: CudaArray,
    out_Pose_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def Pose_update_Mp(
    Pose_r_k: CudaArray,
    Pose_Mp: CudaArray,
    beta: CudaArray,
    out_Pose_Mp_kp1: CudaArray,
    out_Pose_w: CudaArray,
    problem_size: int
) -> None: ...

def Pose_pred_decrease_times_two(
    Pose_step: CudaArray,
    Pose_precond_diag: CudaArray,
    diag: CudaArray,
    Pose_njtr: CudaArray,
    out_Pose_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_retract(
    SimpleRadialExtraCalib: CudaArray,
    delta: CudaArray,
    out_SimpleRadialExtraCalib_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_start_w(
    SimpleRadialExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialExtraCalib_p: CudaArray,
    out_SimpleRadialExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_start_w_contribute(
    SimpleRadialExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialExtraCalib_p: CudaArray,
    out_SimpleRadialExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_alpha_numerator_denominator(
    SimpleRadialExtraCalib_p_kp1: CudaArray,
    SimpleRadialExtraCalib_r_k: CudaArray,
    SimpleRadialExtraCalib_w: CudaArray,
    SimpleRadialExtraCalib_total_ag: CudaArray,
    SimpleRadialExtraCalib_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator(
    SimpleRadialExtraCalib_p_kp1: CudaArray,
    SimpleRadialExtraCalib_w: CudaArray,
    SimpleRadialExtraCalib_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_r_first(
    SimpleRadialExtraCalib_r_k: CudaArray,
    SimpleRadialExtraCalib_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialExtraCalib_r_kp1: CudaArray,
    out_SimpleRadialExtraCalib_r_0_norm2_tot: CudaArray,
    out_SimpleRadialExtraCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_r(
    SimpleRadialExtraCalib_r_k: CudaArray,
    SimpleRadialExtraCalib_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialExtraCalib_r_kp1: CudaArray,
    out_SimpleRadialExtraCalib_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_step_first(
    SimpleRadialExtraCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialExtraCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_step(
    SimpleRadialExtraCalib_step_k: CudaArray,
    SimpleRadialExtraCalib_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialExtraCalib_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_p(
    SimpleRadialExtraCalib_z: CudaArray,
    SimpleRadialExtraCalib_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialExtraCalib_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_update_Mp(
    SimpleRadialExtraCalib_r_k: CudaArray,
    SimpleRadialExtraCalib_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialExtraCalib_Mp_kp1: CudaArray,
    out_SimpleRadialExtraCalib_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialExtraCalib_pred_decrease_times_two(
    SimpleRadialExtraCalib_step: CudaArray,
    SimpleRadialExtraCalib_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialExtraCalib_njtr: CudaArray,
    out_SimpleRadialExtraCalib_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_retract(
    SimpleRadialFocal: CudaArray,
    delta: CudaArray,
    out_SimpleRadialFocal_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_start_w(
    SimpleRadialFocal_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocal_p: CudaArray,
    out_SimpleRadialFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_start_w_contribute(
    SimpleRadialFocal_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocal_p: CudaArray,
    out_SimpleRadialFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_alpha_numerator_denominator(
    SimpleRadialFocal_p_kp1: CudaArray,
    SimpleRadialFocal_r_k: CudaArray,
    SimpleRadialFocal_w: CudaArray,
    SimpleRadialFocal_total_ag: CudaArray,
    SimpleRadialFocal_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_alpha_denumerator_or_beta_nummerator(
    SimpleRadialFocal_p_kp1: CudaArray,
    SimpleRadialFocal_w: CudaArray,
    SimpleRadialFocal_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_r_first(
    SimpleRadialFocal_r_k: CudaArray,
    SimpleRadialFocal_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocal_r_kp1: CudaArray,
    out_SimpleRadialFocal_r_0_norm2_tot: CudaArray,
    out_SimpleRadialFocal_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_r(
    SimpleRadialFocal_r_k: CudaArray,
    SimpleRadialFocal_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocal_r_kp1: CudaArray,
    out_SimpleRadialFocal_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_step_first(
    SimpleRadialFocal_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocal_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_step(
    SimpleRadialFocal_step_k: CudaArray,
    SimpleRadialFocal_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocal_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_p(
    SimpleRadialFocal_z: CudaArray,
    SimpleRadialFocal_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocal_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_update_Mp(
    SimpleRadialFocal_r_k: CudaArray,
    SimpleRadialFocal_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocal_Mp_kp1: CudaArray,
    out_SimpleRadialFocal_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocal_pred_decrease_times_two(
    SimpleRadialFocal_step: CudaArray,
    SimpleRadialFocal_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocal_njtr: CudaArray,
    out_SimpleRadialFocal_pred_dec: CudaArray,
    problem_size: int
) -> None: ...

