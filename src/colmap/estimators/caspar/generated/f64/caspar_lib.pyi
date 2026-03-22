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
                 Point_num_max: int = 0,
                 Pose_num_max: int = 0,
                 SimpleRadialCalib_num_max: int = 0,
                 simple_radial_num_max: int = 0,
                 simple_radial_fixed_pose_num_max: int = 0,
                 simple_radial_fixed_point_num_max: int = 0,
                 simple_radial_fixed_calib_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_calib_num_max: int = 0,
                 simple_radial_fixed_point_fixed_calib_num_max: int = 0,
                 pinhole_num_max: int = 0,
                 pinhole_fixed_pose_num_max: int = 0,
                 pinhole_fixed_point_num_max: int = 0,
                 pinhole_fixed_calib_num_max: int = 0,
                 pinhole_fixed_pose_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_calib_num_max: int = 0,
                 pinhole_fixed_point_fixed_calib_num_max: int = 0,
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
    def set_simple_radial_fixed_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_calib factor from host.
        """

    def set_simple_radial_fixed_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_calib factor from device.
        """
    def set_simple_radial_fixed_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_calib factor from host.
        """

    def set_simple_radial_fixed_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_calib factor from device.
        """

    def set_simple_radial_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_calib factors.

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
    def set_simple_radial_fixed_pose_fixed_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_calib factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_calib factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_point_fixed_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_point_fixed_calib factor from host.
        """

    def set_simple_radial_fixed_point_fixed_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_point_fixed_calib factor from device.
        """

    def set_simple_radial_fixed_point_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_point_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts simple_radial_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_point_fixed_calib_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_fixed_calib_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_point_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_point_fixed_calib factors.

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
    def set_pinhole_fixed_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_calib factor from host.
        """

    def set_pinhole_fixed_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_calib factor from device.
        """
    def set_pinhole_fixed_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_calib factor from host.
        """

    def set_pinhole_fixed_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_calib factor from device.
        """

    def set_pinhole_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_calib factors.

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
    def set_pinhole_fixed_pose_fixed_calib_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_calib factor from host.
        """

    def set_pinhole_fixed_pose_fixed_calib_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_calib factor from device.
        """

    def set_pinhole_fixed_pose_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_calib_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_calib_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_pose_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_pose_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_point_fixed_calib_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_point_fixed_calib factor from host.
        """

    def set_pinhole_fixed_point_fixed_calib_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_point_fixed_calib factor from device.
        """

    def set_pinhole_fixed_point_fixed_calib_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_fixed_calib_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_point_fixed_calib_calib_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_fixed_calib_calib_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the calib consts pinhole_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_point_fixed_calib_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_point_fixed_calib factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_fixed_calib_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_point_fixed_calib factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_point_fixed_calib_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_point_fixed_calib factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

def ConstPinholeCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholeCalib data to the caspar data format.
    """

def ConstPinholeCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholeCalib data to the stacked data format.
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

def ConstSimpleRadialCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialCalib data to the caspar data format.
    """

def ConstSimpleRadialCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialCalib data to the stacked data format.
    """

def PinholeCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeCalib data to the caspar data format.
    """

def PinholeCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeCalib data to the stacked data format.
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

def SimpleRadialCalib_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialCalib data to the caspar data format.
    """

def SimpleRadialCalib_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialCalib data to the stacked data format.
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

def simple_radial_fixed_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
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

def simple_radial_fixed_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
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

def simple_radial_fixed_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_calib_jtjnjtr_direct(
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

def simple_radial_fixed_pose_fixed_calib_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_calib_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_calib_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_calib_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_fixed_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_fixed_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_fixed_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_fixed_calib_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
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

def pinhole_fixed_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
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

def pinhole_fixed_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
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

def pinhole_fixed_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_calib_jtjnjtr_direct(
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

def pinhole_fixed_pose_fixed_calib_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_calib_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_calib_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    calib: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_calib_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_fixed_calib_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_fixed_calib_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_fixed_calib_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    calib: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_fixed_calib_jtjnjtr_direct(
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

def PinholeCalib_alpha_denumerator_or_beta_nummerator(
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

def SimpleRadialCalib_alpha_denumerator_or_beta_nummerator(
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

