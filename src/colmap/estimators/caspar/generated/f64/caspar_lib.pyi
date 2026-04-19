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
                 PinholeFocalAndExtra_num_max: int = 0,
                 PinholePrincipalPoint_num_max: int = 0,
                 Point_num_max: int = 0,
                 Pose_num_max: int = 0,
                 SimpleRadialFocalAndExtra_num_max: int = 0,
                 SimpleRadialPrincipalPoint_num_max: int = 0,
                 simple_radial_num_max: int = 0,
                 simple_radial_fixed_pose_num_max: int = 0,
                 simple_radial_fixed_focal_and_extra_num_max: int = 0,
                 simple_radial_fixed_principal_point_num_max: int = 0,
                 simple_radial_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_and_extra_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_principal_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_point_num_max: int = 0,
                 simple_radial_fixed_focal_and_extra_fixed_principal_point_num_max: int = 0,
                 simple_radial_fixed_focal_and_extra_fixed_point_num_max: int = 0,
                 simple_radial_fixed_principal_point_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num_max: int = 0,
                 simple_radial_fixed_pose_fixed_principal_point_fixed_point_num_max: int = 0,
                 simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_num_max: int = 0,
                 pinhole_fixed_pose_num_max: int = 0,
                 pinhole_fixed_focal_and_extra_num_max: int = 0,
                 pinhole_fixed_principal_point_num_max: int = 0,
                 pinhole_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_and_extra_num_max: int = 0,
                 pinhole_fixed_pose_fixed_principal_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_point_num_max: int = 0,
                 pinhole_fixed_focal_and_extra_fixed_principal_point_num_max: int = 0,
                 pinhole_fixed_focal_and_extra_fixed_point_num_max: int = 0,
                 pinhole_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max: int = 0,
                 pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max: int = 0,
                 pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num_max: int = 0,
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

    def set_PinholeFocalAndExtra_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the PinholeFocalAndExtra nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_PinholeFocalAndExtra_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the PinholeFocalAndExtra nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_PinholeFocalAndExtra_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the PinholeFocalAndExtra nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_PinholeFocalAndExtra_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the PinholeFocalAndExtra nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_PinholeFocalAndExtra_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type PinholeFocalAndExtra.

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
    def set_SimpleRadialFocalAndExtra_nodes_from_stacked_host(self, stacked_data: Array, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocalAndExtra nodes from the stacked host data.

        The offset can be used to start writing at a specific index.
        """

    def set_SimpleRadialFocalAndExtra_nodes_from_stacked_device(self, stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Set the current value for the SimpleRadialFocalAndExtra nodes from the stacked device data.

        The offset can be used to start writing at a specific index.
        """

    def get_SimpleRadialFocalAndExtra_nodes_to_stacked_host(self, out_stacked_data: Array, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocalAndExtra nodes into the stacked output host data.

        The offset can be used to start reading from a specific index.
        """

    def get_SimpleRadialFocalAndExtra_nodes_to_stacked_device(self, out_stacked_data: CudaArray, offset: int = 0) -> None:
        """
        Read the current value for the SimpleRadialFocalAndExtra nodes into the stacked output device data.

        The offset can be used to start reading from a specific index.
        """

    def set_SimpleRadialFocalAndExtra_num(self, num: int) -> None:
        """
        Set the current number of active nodes of type SimpleRadialFocalAndExtra.

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
    def set_simple_radial_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial factor from host.
        """

    def set_simple_radial_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial factor from device.
        """
    def set_simple_radial_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial factor from host.
        """

    def set_simple_radial_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial factor from device.
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
    def set_simple_radial_fixed_pose_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose factor from device.
        """
    def set_simple_radial_fixed_pose_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose factor from host.
        """

    def set_simple_radial_fixed_pose_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose factor from device.
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
    def set_simple_radial_fixed_focal_and_extra_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra factor from device.
        """
    def set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_focal_and_extra factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_focal_and_extra factor from device.
        """
    def set_simple_radial_fixed_focal_and_extra_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_and_extra factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_and_extra factor from device.
        """

    def set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_and_extra factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_principal_point factor from device.
        """
    def set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_principal_point factor from device.
        """
    def set_simple_radial_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_principal_point factor from device.
        """

    def set_simple_radial_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_principal_point factors.

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
    def set_simple_radial_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_point factor from device.
        """
    def set_simple_radial_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_point factor from host.
        """

    def set_simple_radial_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_point factor from device.
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
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_focal_and_extra factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_focal_and_extra factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_and_extra factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_and_extra factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal_and_extra factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_principal_point factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_principal_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_point factor from device.
        """
    def set_simple_radial_fixed_pose_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_point factor from device.
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
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point factor from device.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point factor from device.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_and_extra_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_point factor from device.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_focal_and_extra_fixed_point factor from device.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_and_extra_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_principal_point_fixed_point factor from device.
        """
    def set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_pose_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from host.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from device.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point factors.

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
    def set_pinhole_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole factor from host.
        """

    def set_pinhole_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole factor from device.
        """
    def set_pinhole_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole factor from host.
        """

    def set_pinhole_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole factor from device.
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
    def set_pinhole_fixed_pose_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose factor from device.
        """
    def set_pinhole_fixed_pose_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose factor from host.
        """

    def set_pinhole_fixed_pose_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose factor from device.
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
    def set_pinhole_fixed_focal_and_extra_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra factor from device.
        """
    def set_pinhole_fixed_focal_and_extra_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_focal_and_extra factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_focal_and_extra factor from device.
        """
    def set_pinhole_fixed_focal_and_extra_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_and_extra factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_and_extra factor from device.
        """

    def set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_and_extra factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_principal_point factor from device.
        """
    def set_pinhole_fixed_principal_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_principal_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_principal_point factor from device.
        """
    def set_pinhole_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_principal_point factor from device.
        """

    def set_pinhole_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_principal_point factors.

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
    def set_pinhole_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_point factor from device.
        """
    def set_pinhole_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_point factor from host.
        """

    def set_pinhole_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_point factor from device.
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
    def set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_focal_and_extra factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_focal_and_extra factor from device.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_and_extra factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_and_extra factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal_and_extra factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_principal_point factor from device.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_principal_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_point factor from device.
        """
    def set_pinhole_fixed_pose_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_point factor from device.
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
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_principal_point factor from device.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_focal_and_extra_fixed_principal_point factor from device.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_and_extra_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_point factor from device.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_focal_and_extra_fixed_point factor from device.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_and_extra_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_principal_point_fixed_point factor from device.
        """
    def set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the point argument for the pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the principal_point argument for the pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_focal_and_extra_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the focal_and_extra argument for the pinhole_fixed_pose_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pose consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_pose_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_pose_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(self, indices: Array) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from host.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(self, indices: CudaArray) -> None:
        """
        Set the indices for the pose argument for the pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from device.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the pixel consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the focal_and_extra consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the principal_point consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """
    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked host data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
        self, stacked_data: Array, offset: int = 0
        ) -> None:
        """
        Set the values for the point consts pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factor from stacked device data.

        The offset can be used to start writing from a specific index.
        """

    def set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num(self, num: int) -> None:
        """
        Set the current number of pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point factors.

        The value is set during initialization and this function is only needed if you want to change
        the problem between optimization runs. This is work in progress and can have performance impacts.
        """

def ConstPinholeFocalAndExtra_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholeFocalAndExtra data to the caspar data format.
    """

def ConstPinholeFocalAndExtra_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholeFocalAndExtra data to the stacked data format.
    """

def ConstPinholePrincipalPoint_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstPinholePrincipalPoint data to the caspar data format.
    """

def ConstPinholePrincipalPoint_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstPinholePrincipalPoint data to the stacked data format.
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

def ConstSimpleRadialFocalAndExtra_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialFocalAndExtra data to the caspar data format.
    """

def ConstSimpleRadialFocalAndExtra_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialFocalAndExtra data to the stacked data format.
    """

def ConstSimpleRadialPrincipalPoint_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked ConstSimpleRadialPrincipalPoint data to the caspar data format.
    """

def ConstSimpleRadialPrincipalPoint_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar ConstSimpleRadialPrincipalPoint data to the stacked data format.
    """

def PinholeFocalAndExtra_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholeFocalAndExtra data to the caspar data format.
    """

def PinholeFocalAndExtra_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholeFocalAndExtra data to the stacked data format.
    """

def PinholePrincipalPoint_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked PinholePrincipalPoint data to the caspar data format.
    """

def PinholePrincipalPoint_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar PinholePrincipalPoint data to the stacked data format.
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

def SimpleRadialFocalAndExtra_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialFocalAndExtra data to the caspar data format.
    """

def SimpleRadialFocalAndExtra_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
    """
    Convert the caspar SimpleRadialFocalAndExtra data to the stacked data format.
    """

def SimpleRadialPrincipalPoint_stacked_to_caspar(stacked_data: CudaArray, out_cas_data: CudaArray) -> None:
    """
    Convert the stacked SimpleRadialPrincipalPoint data to the caspar data format.
    """

def SimpleRadialPrincipalPoint_caspar_to_stacked(caspar_data: CudaArray, out_stacked_data: CudaArray) -> None:
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
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def simple_radial_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def simple_radial_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
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
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def simple_radial_fixed_pose_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def simple_radial_fixed_pose_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_jtjnjtr_direct(
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

def simple_radial_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
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
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_pose_fixed_focal_and_extra_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_pose_fixed_focal_and_extra_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct(
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

def simple_radial_fixed_pose_fixed_principal_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
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

def simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def simple_radial_fixed_focal_and_extra_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
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

def simple_radial_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_fixed_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_fixed_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def pinhole_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def pinhole_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
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
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def pinhole_fixed_pose_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
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

def pinhole_fixed_pose_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_jtjnjtr_direct(
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

def pinhole_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
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
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_pose_fixed_focal_and_extra_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_pose_fixed_focal_and_extra_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct(
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

def pinhole_fixed_pose_fixed_principal_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_point_jac: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    out_principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_fixed_principal_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_fixed_principal_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
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

def pinhole_fixed_focal_and_extra_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
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

def pinhole_fixed_focal_and_extra_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
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

def pinhole_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    out_focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    out_pose_njtr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_res: CudaArray,
    out_point_njtr: CudaArray,
    out_point_precond_diag: CudaArray,
    out_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score(
    point: CudaArray,
    point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_jtjnjtr_direct(
    point_njtr: CudaArray,
    point_njtr_indices: CudaArray,
    point_jac: CudaArray,
    out_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_principal_point_njtr: CudaArray,
    out_principal_point_precond_diag: CudaArray,
    out_principal_point_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_score(
    principal_point: CudaArray,
    principal_point_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    focal_and_extra: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
    principal_point_njtr: CudaArray,
    principal_point_njtr_indices: CudaArray,
    principal_point_jac: CudaArray,
    out_principal_point_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac_first(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_fixed_point_res_jac(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    out_focal_and_extra_precond_diag: CudaArray,
    out_focal_and_extra_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_fixed_point_score(
    focal_and_extra: CudaArray,
    focal_and_extra_indices: CudaArray,
    pixel: CudaArray,
    pose: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_pose_fixed_principal_point_fixed_point_jtjnjtr_direct(
    focal_and_extra_njtr: CudaArray,
    focal_and_extra_njtr_indices: CudaArray,
    focal_and_extra_jac: CudaArray,
    out_focal_and_extra_njtr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_rTr: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_res: CudaArray,
    out_pose_njtr: CudaArray,
    out_pose_precond_diag: CudaArray,
    out_pose_precond_tril: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_score(
    pose: CudaArray,
    pose_indices: CudaArray,
    pixel: CudaArray,
    focal_and_extra: CudaArray,
    principal_point: CudaArray,
    point: CudaArray,
    out_rTr: CudaArray,
    problem_size: int
) -> None: ...

def pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_jtjnjtr_direct(
    pose_njtr: CudaArray,
    pose_njtr_indices: CudaArray,
    pose_jac: CudaArray,
    out_pose_njtr: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_retract(
    PinholeFocalAndExtra: CudaArray,
    delta: CudaArray,
    out_PinholeFocalAndExtra_retracted: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_start_w(
    PinholeFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocalAndExtra_p: CudaArray,
    out_PinholeFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_start_w_contribute(
    PinholeFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocalAndExtra_p: CudaArray,
    out_PinholeFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_alpha_numerator_denominator(
    PinholeFocalAndExtra_p_kp1: CudaArray,
    PinholeFocalAndExtra_r_k: CudaArray,
    PinholeFocalAndExtra_w: CudaArray,
    PinholeFocalAndExtra_total_ag: CudaArray,
    PinholeFocalAndExtra_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_alpha_denumerator_or_beta_nummerator(
    PinholeFocalAndExtra_p_kp1: CudaArray,
    PinholeFocalAndExtra_w: CudaArray,
    PinholeFocalAndExtra_out: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_r_first(
    PinholeFocalAndExtra_r_k: CudaArray,
    PinholeFocalAndExtra_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeFocalAndExtra_r_kp1: CudaArray,
    out_PinholeFocalAndExtra_r_0_norm2_tot: CudaArray,
    out_PinholeFocalAndExtra_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_r(
    PinholeFocalAndExtra_r_k: CudaArray,
    PinholeFocalAndExtra_w: CudaArray,
    negalpha: CudaArray,
    out_PinholeFocalAndExtra_r_kp1: CudaArray,
    out_PinholeFocalAndExtra_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_step_first(
    PinholeFocalAndExtra_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeFocalAndExtra_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_step(
    PinholeFocalAndExtra_step_k: CudaArray,
    PinholeFocalAndExtra_p_kp1: CudaArray,
    alpha: CudaArray,
    out_PinholeFocalAndExtra_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_p(
    PinholeFocalAndExtra_z: CudaArray,
    PinholeFocalAndExtra_p_k: CudaArray,
    beta: CudaArray,
    out_PinholeFocalAndExtra_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_update_Mp(
    PinholeFocalAndExtra_r_k: CudaArray,
    PinholeFocalAndExtra_Mp: CudaArray,
    beta: CudaArray,
    out_PinholeFocalAndExtra_Mp_kp1: CudaArray,
    out_PinholeFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def PinholeFocalAndExtra_pred_decrease_times_two(
    PinholeFocalAndExtra_step: CudaArray,
    PinholeFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    PinholeFocalAndExtra_njtr: CudaArray,
    out_PinholeFocalAndExtra_pred_dec: CudaArray,
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

def PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator(
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

def SimpleRadialFocalAndExtra_retract(
    SimpleRadialFocalAndExtra: CudaArray,
    delta: CudaArray,
    out_SimpleRadialFocalAndExtra_retracted: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_normalize(
    precond_diag: CudaArray,
    precond_tril: CudaArray,
    njtr: CudaArray,
    diag: CudaArray,
    out_normalized: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_start_w(
    SimpleRadialFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndExtra_p: CudaArray,
    out_SimpleRadialFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_start_w_contribute(
    SimpleRadialFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndExtra_p: CudaArray,
    out_SimpleRadialFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_alpha_numerator_denominator(
    SimpleRadialFocalAndExtra_p_kp1: CudaArray,
    SimpleRadialFocalAndExtra_r_k: CudaArray,
    SimpleRadialFocalAndExtra_w: CudaArray,
    SimpleRadialFocalAndExtra_total_ag: CudaArray,
    SimpleRadialFocalAndExtra_total_ac: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator(
    SimpleRadialFocalAndExtra_p_kp1: CudaArray,
    SimpleRadialFocalAndExtra_w: CudaArray,
    SimpleRadialFocalAndExtra_out: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_r_first(
    SimpleRadialFocalAndExtra_r_k: CudaArray,
    SimpleRadialFocalAndExtra_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocalAndExtra_r_kp1: CudaArray,
    out_SimpleRadialFocalAndExtra_r_0_norm2_tot: CudaArray,
    out_SimpleRadialFocalAndExtra_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_r(
    SimpleRadialFocalAndExtra_r_k: CudaArray,
    SimpleRadialFocalAndExtra_w: CudaArray,
    negalpha: CudaArray,
    out_SimpleRadialFocalAndExtra_r_kp1: CudaArray,
    out_SimpleRadialFocalAndExtra_r_kp1_norm2_tot: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_step_first(
    SimpleRadialFocalAndExtra_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocalAndExtra_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_step(
    SimpleRadialFocalAndExtra_step_k: CudaArray,
    SimpleRadialFocalAndExtra_p_kp1: CudaArray,
    alpha: CudaArray,
    out_SimpleRadialFocalAndExtra_step_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_p(
    SimpleRadialFocalAndExtra_z: CudaArray,
    SimpleRadialFocalAndExtra_p_k: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocalAndExtra_p_kp1: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_update_Mp(
    SimpleRadialFocalAndExtra_r_k: CudaArray,
    SimpleRadialFocalAndExtra_Mp: CudaArray,
    beta: CudaArray,
    out_SimpleRadialFocalAndExtra_Mp_kp1: CudaArray,
    out_SimpleRadialFocalAndExtra_w: CudaArray,
    problem_size: int
) -> None: ...

def SimpleRadialFocalAndExtra_pred_decrease_times_two(
    SimpleRadialFocalAndExtra_step: CudaArray,
    SimpleRadialFocalAndExtra_precond_diag: CudaArray,
    diag: CudaArray,
    SimpleRadialFocalAndExtra_njtr: CudaArray,
    out_SimpleRadialFocalAndExtra_pred_dec: CudaArray,
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

def SimpleRadialPrincipalPoint_alpha_denumerator_or_beta_nummerator(
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

