#include "pybind_array_tools.h"
#include "solver.h"
#include <pybind11/stl.h>

namespace caspar {

inline void add_solver_pybinding(pybind11::module_ module) {
  py::enum_<ExitReason>(module, "ExitReason")
      .value("MAX_ITERATIONS", ExitReason::MAX_ITERATIONS)
      .value("CONVERGED_SCORE_THRESHOLD", ExitReason::CONVERGED_SCORE_THRESHOLD)
      .value("CONVERGED_DIAG_EXIT", ExitReason::CONVERGED_DIAG_EXIT)
      .export_values();

  py::class_<IterationData>(module, "IterationData")
      .def(py::init<>())
      .def_readwrite("solver_iter", &IterationData::solver_iter)
      .def_readwrite("pcg_iter", &IterationData::pcg_iter)
      .def_readwrite("score_current", &IterationData::score_current)
      .def_readwrite("score_best", &IterationData::score_best)
      .def_readwrite("step_quality", &IterationData::step_quality)
      .def_readwrite("diag", &IterationData::diag)
      .def_readwrite("dt_inc", &IterationData::dt_inc)
      .def_readwrite("dt_tot", &IterationData::dt_tot)
      .def_readwrite("step_accepted", &IterationData::step_accepted);

  py::class_<SolveResult>(module, "SolveResult")
      .def(py::init<>())
      .def_readwrite("initial_score", &SolveResult::initial_score)
      .def_readwrite("final_score", &SolveResult::final_score)
      .def_readwrite("iteration_count", &SolveResult::iteration_count)
      .def_readwrite("runtime", &SolveResult::runtime)
      .def_readwrite("exit_reason", &SolveResult::exit_reason)
      .def_readwrite("iterations", &SolveResult::iterations);
  py::class_<GraphSolver>(
      module, "GraphSolver", "Class for solving Factor Graphs.")
      .def(
          py::init<SolverParams<double>,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t,
                   size_t>(),
          py::arg("params"),
          py::kw_only(),
          py::arg("PinholeCalib_num_max") = 0,
          py::arg("PinholeFocalAndExtra_num_max") = 0,
          py::arg("PinholePose_num_max") = 0,
          py::arg("PinholePrincipalPoint_num_max") = 0,
          py::arg("Point_num_max") = 0,
          py::arg("SimpleRadialCalib_num_max") = 0,
          py::arg("SimpleRadialFocalAndExtra_num_max") = 0,
          py::arg("SimpleRadialPose_num_max") = 0,
          py::arg("SimpleRadialPrincipalPoint_num_max") = 0,
          py::arg("simple_radial_merged_num_max") = 0,
          py::arg("simple_radial_merged_fixed_pose_num_max") = 0,
          py::arg("simple_radial_merged_fixed_point_num_max") = 0,
          py::arg("simple_radial_merged_fixed_pose_fixed_point_num_max") = 0,
          py::arg("pinhole_merged_num_max") = 0,
          py::arg("pinhole_merged_fixed_pose_num_max") = 0,
          py::arg("pinhole_merged_fixed_point_num_max") = 0,
          py::arg("pinhole_merged_fixed_pose_fixed_point_num_max") = 0,
          py::arg("simple_radial_fixed_focal_and_extra_num_max") = 0,
          py::arg("simple_radial_fixed_principal_point_num_max") = 0,
          py::arg("simple_radial_fixed_pose_fixed_focal_and_extra_num_max") = 0,
          py::arg("simple_radial_fixed_pose_fixed_principal_point_num_max") = 0,
          py::arg("simple_radial_fixed_focal_and_extra_fixed_principal_point_"
                  "num_max") = 0,
          py::arg("simple_radial_fixed_focal_and_extra_fixed_point_num_max") =
              0,
          py::arg("simple_radial_fixed_principal_point_fixed_point_num_max") =
              0,
          py::arg("simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
                  "principal_point_num_max") = 0,
          py::arg("simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
                  "num_max") = 0,
          py::arg("simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
                  "num_max") = 0,
          py::arg("simple_radial_fixed_focal_and_extra_fixed_principal_point_"
                  "fixed_point_num_max") = 0,
          py::arg("pinhole_fixed_focal_and_extra_num_max") = 0,
          py::arg("pinhole_fixed_principal_point_num_max") = 0,
          py::arg("pinhole_fixed_pose_fixed_focal_and_extra_num_max") = 0,
          py::arg("pinhole_fixed_pose_fixed_principal_point_num_max") = 0,
          py::arg(
              "pinhole_fixed_focal_and_extra_fixed_principal_point_num_max") =
              0,
          py::arg("pinhole_fixed_focal_and_extra_fixed_point_num_max") = 0,
          py::arg("pinhole_fixed_principal_point_fixed_point_num_max") = 0,
          py::arg("pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
                  "point_num_max") = 0,
          py::arg(
              "pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num_max") =
              0,
          py::arg(
              "pinhole_fixed_pose_fixed_principal_point_fixed_point_num_max") =
              0,
          py::arg("pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
                  "point_num_max") = 0)

      .def("set_params", &GraphSolver::set_params)
      .def("solve",
           &GraphSolver::solve,
           py::call_guard<py::gil_scoped_release>(),
           py::arg("print_progress") = false,
           py::arg("verbose_logging") = false)
      .def("finish_indices", &GraphSolver::finish_indices)
      .def("get_allocation_size", &GraphSolver::get_allocation_size)

      .def("set_PinholeCalib_num", &GraphSolver::set_PinholeCalib_num)
      .def(
          "set_PinholeCalib_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_PinholeCalib_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_PinholeCalib_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_PinholeCalib_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholeCalib_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_PinholeCalib_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholeCalib_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_PinholeCalib_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_PinholeFocalAndExtra_num",
           &GraphSolver::set_PinholeFocalAndExtra_num)
      .def(
          "set_PinholeFocalAndExtra_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_PinholeFocalAndExtra_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_PinholeFocalAndExtra_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_PinholeFocalAndExtra_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholeFocalAndExtra_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_PinholeFocalAndExtra_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholeFocalAndExtra_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_PinholeFocalAndExtra_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_PinholePose_num", &GraphSolver::set_PinholePose_num)
      .def(
          "set_PinholePose_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_PinholePose_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_PinholePose_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_PinholePose_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholePose_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_PinholePose_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholePose_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_PinholePose_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_PinholePrincipalPoint_num",
           &GraphSolver::set_PinholePrincipalPoint_num)
      .def(
          "set_PinholePrincipalPoint_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_PinholePrincipalPoint_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_PinholePrincipalPoint_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_PinholePrincipalPoint_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholePrincipalPoint_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_PinholePrincipalPoint_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_PinholePrincipalPoint_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_PinholePrincipalPoint_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_Point_num", &GraphSolver::set_Point_num)
      .def(
          "set_Point_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_Point_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_Point_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_Point_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_Point_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_Point_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_Point_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_Point_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_SimpleRadialCalib_num", &GraphSolver::set_SimpleRadialCalib_num)
      .def(
          "set_SimpleRadialCalib_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_SimpleRadialCalib_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_SimpleRadialCalib_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_SimpleRadialCalib_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialCalib_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_SimpleRadialCalib_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialCalib_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_SimpleRadialCalib_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_SimpleRadialFocalAndExtra_num",
           &GraphSolver::set_SimpleRadialFocalAndExtra_num)
      .def(
          "set_SimpleRadialFocalAndExtra_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_SimpleRadialFocalAndExtra_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_SimpleRadialFocalAndExtra_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_SimpleRadialFocalAndExtra_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialFocalAndExtra_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_SimpleRadialFocalAndExtra_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialFocalAndExtra_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_SimpleRadialFocalAndExtra_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_SimpleRadialPose_num", &GraphSolver::set_SimpleRadialPose_num)
      .def(
          "set_SimpleRadialPose_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_SimpleRadialPose_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_SimpleRadialPose_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_SimpleRadialPose_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialPose_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_SimpleRadialPose_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialPose_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_SimpleRadialPose_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_SimpleRadialPrincipalPoint_num",
           &GraphSolver::set_SimpleRadialPrincipalPoint_num)
      .def(
          "set_SimpleRadialPrincipalPoint_nodes_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_SimpleRadialPrincipalPoint_nodes_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "set_SimpleRadialPrincipalPoint_nodes_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_SimpleRadialPrincipalPoint_nodes_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialPrincipalPoint_nodes_to_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.get_SimpleRadialPrincipalPoint_nodes_to_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)
      .def(
          "get_SimpleRadialPrincipalPoint_nodes_to_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.get_SimpleRadialPrincipalPoint_nodes_to_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_nodes"),
          pybind11::arg("offset") = 0)

      .def("set_simple_radial_merged_num",
           &GraphSolver::set_simple_radial_merged_num)

      .def("set_simple_radial_merged_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_simple_radial_merged_pose_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_calib_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_calib_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_calib_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_simple_radial_merged_calib_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_simple_radial_merged_point_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_simple_radial_merged_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_simple_radial_merged_pixel_data_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_simple_radial_merged_pixel_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_merged_fixed_pose_num",
           &GraphSolver::set_simple_radial_merged_fixed_pose_num)

      .def("set_simple_radial_merged_fixed_pose_calib_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_fixed_pose_calib_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_fixed_pose_calib_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver
                 .set_simple_radial_merged_fixed_pose_calib_indices_from_device(
                     AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_fixed_pose_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_fixed_pose_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_fixed_pose_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver
                 .set_simple_radial_merged_fixed_pose_point_indices_from_device(
                     AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_pose_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_pose_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_merged_fixed_point_num",
           &GraphSolver::set_simple_radial_merged_fixed_point_num)

      .def("set_simple_radial_merged_fixed_point_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_simple_radial_merged_fixed_point_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_simple_radial_merged_fixed_point_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver
                 .set_simple_radial_merged_fixed_point_pose_indices_from_device(
                     AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_simple_radial_merged_fixed_point_calib_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver.set_simple_radial_merged_fixed_point_calib_indices_from_host(
                AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_merged_fixed_point_calib_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_merged_fixed_point_calib_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_merged_fixed_point_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_point_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_point_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_point_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_merged_fixed_pose_fixed_point_num",
           &GraphSolver::set_simple_radial_merged_fixed_pose_fixed_point_num)

      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_from_"
          "host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_pose_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_pose_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_merged_fixed_pose_fixed_point_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_merged_fixed_pose_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_merged_num", &GraphSolver::set_pinhole_merged_num)

      .def("set_pinhole_merged_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_pose_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_calib_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_calib_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_calib_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_calib_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_point_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_merged_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_pinhole_merged_pixel_data_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_pinhole_merged_pixel_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_merged_fixed_pose_num",
           &GraphSolver::set_pinhole_merged_fixed_pose_num)

      .def("set_pinhole_merged_fixed_pose_calib_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_fixed_pose_calib_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_pose_calib_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_fixed_pose_calib_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_pose_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_fixed_pose_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_pose_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_fixed_pose_point_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_merged_fixed_pose_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_pinhole_merged_fixed_pose_pixel_data_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_pinhole_merged_fixed_pose_pixel_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_pose_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver.set_pinhole_merged_fixed_pose_pose_data_from_stacked_device(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_pose_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_pinhole_merged_fixed_pose_pose_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_merged_fixed_point_num",
           &GraphSolver::set_pinhole_merged_fixed_point_num)

      .def("set_pinhole_merged_fixed_point_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_fixed_point_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_point_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_fixed_point_pose_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_point_calib_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_merged_fixed_point_calib_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_merged_fixed_point_calib_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_merged_fixed_point_calib_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_merged_fixed_point_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_point_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_pinhole_merged_fixed_point_pixel_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_point_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_point_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver.set_pinhole_merged_fixed_point_point_data_from_stacked_host(
                AsDoublePtr(stacked_data), offset, GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_merged_fixed_pose_fixed_point_num",
           &GraphSolver::set_pinhole_merged_fixed_pose_fixed_point_num)

      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_point_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_merged_fixed_pose_fixed_point_point_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_merged_fixed_pose_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_fixed_focal_and_extra_num",
           &GraphSolver::set_simple_radial_fixed_focal_and_extra_num)

      .def(
          "set_simple_radial_fixed_focal_and_extra_pose_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_pose_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_principal_point_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_principal_point_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_fixed_principal_point_num",
           &GraphSolver::set_simple_radial_fixed_principal_point_num)

      .def(
          "set_simple_radial_fixed_principal_point_pose_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_pose_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_focal_and_extra_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_focal_and_extra_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_pixel_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_pixel_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_principal_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_principal_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_fixed_pose_fixed_focal_and_extra_num",
           &GraphSolver::set_simple_radial_fixed_pose_fixed_focal_and_extra_num)

      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_simple_radial_fixed_pose_fixed_principal_point_num",
           &GraphSolver::set_simple_radial_fixed_pose_fixed_principal_point_num)

      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_point_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_point_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_principal_point_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_principal_point_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num",
          &GraphSolver::
              set_simple_radial_fixed_focal_and_extra_fixed_principal_point_num)

      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_"
          "and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_"
          "and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
          "principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
          "principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_num",
          &GraphSolver::set_simple_radial_fixed_focal_and_extra_fixed_point_num)

      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_num",
          &GraphSolver::set_simple_radial_fixed_principal_point_fixed_point_num)

      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_pose_indices_"
          "from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_pose_indices_"
          "from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_principal_point_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_principal_point_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_principal_point_fixed_point_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_num",
          &GraphSolver::
              set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num)

      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_pose_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_pose_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_focal_and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_focal_and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_"
          "point_principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num",
          &GraphSolver::
              set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_num)

      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "principal_point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "principal_point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "focal_and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "focal_and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
          "point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num",
          &GraphSolver::
              set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_num)

      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "focal_and_extra_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "focal_and_extra_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_"
          "data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_"
          "data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
          "point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_num",
          &GraphSolver::
              set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_num)

      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_pose_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_pose_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_focal_and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_focal_and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_"
          "point_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_simple_radial_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_focal_and_extra_num",
           &GraphSolver::set_pinhole_fixed_focal_and_extra_num)

      .def("set_pinhole_fixed_focal_and_extra_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_fixed_focal_and_extra_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_fixed_focal_and_extra_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_fixed_focal_and_extra_pose_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_fixed_focal_and_extra_principal_point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_principal_point_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def("set_pinhole_fixed_focal_and_extra_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_fixed_focal_and_extra_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_fixed_focal_and_extra_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_fixed_focal_and_extra_point_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_principal_point_num",
           &GraphSolver::set_pinhole_fixed_principal_point_num)

      .def("set_pinhole_fixed_principal_point_pose_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_fixed_principal_point_pose_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_fixed_principal_point_pose_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_fixed_principal_point_pose_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_fixed_principal_point_focal_and_extra_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_principal_point_focal_and_extra_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def("set_pinhole_fixed_principal_point_point_indices_from_host",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertHostMemory(indices);
             solver.set_pinhole_fixed_principal_point_point_indices_from_host(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def("set_pinhole_fixed_principal_point_point_indices_from_device",
           [](GraphSolver& solver, pybind11::object indices) {
             AssertDeviceMemory(indices);
             solver.set_pinhole_fixed_principal_point_point_indices_from_device(
                 AsUintPtr(indices), GetNumRows(indices));
           })
      .def(
          "set_pinhole_fixed_principal_point_pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_principal_point_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_principal_point_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_pose_fixed_focal_and_extra_num",
           &GraphSolver::set_pinhole_fixed_pose_fixed_focal_and_extra_num)

      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_"
          "host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_pose_fixed_principal_point_num",
           &GraphSolver::set_pinhole_fixed_pose_fixed_principal_point_num)

      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_"
          "host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_"
          "device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_"
          "host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_focal_and_extra_fixed_principal_point_num",
           &GraphSolver::
               set_pinhole_fixed_focal_and_extra_fixed_principal_point_num)

      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_"
          "extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_"
          "extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_"
          "point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_"
          "point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_focal_and_extra_fixed_point_num",
           &GraphSolver::set_pinhole_fixed_focal_and_extra_fixed_point_num)

      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_"
          "host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_principal_point_fixed_point_num",
           &GraphSolver::set_pinhole_fixed_principal_point_fixed_point_num)

      .def(
          "set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_"
          "host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_"
          "device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_"
          "indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_"
          "indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_principal_point_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_principal_point_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_point_data_from_"
          "stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_principal_point_fixed_point_point_data_from_"
          "stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "num",
          &GraphSolver::
              set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_num)

      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "pose_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "pose_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "focal_and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "focal_and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_"
          "principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num",
           &GraphSolver::
               set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num)

      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_"
          "point_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_"
          "point_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_principal_point_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_"
          "extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_"
          "extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def("set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num",
           &GraphSolver::
               set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num)

      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_"
          "extra_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_"
          "extra_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_and_extra_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_"
          "point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_"
          "point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_"
          "from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_"
          "from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "num",
          &GraphSolver::
              set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_num)

      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "pose_indices_from_host",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertHostMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_host(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "pose_indices_from_device",
          [](GraphSolver& solver, pybind11::object indices) {
            AssertDeviceMemory(indices);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pose_indices_from_device(
                    AsUintPtr(indices), GetNumRows(indices));
          })
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "pixel_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "pixel_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_pixel_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "focal_and_extra_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "focal_and_extra_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_focal_and_extra_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "principal_point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "principal_point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_principal_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "point_data_from_stacked_device",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertDeviceMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_device(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0)
      .def(
          "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_"
          "point_data_from_stacked_host",
          [](GraphSolver& solver,
             pybind11::object stacked_data,
             size_t offset) {
            AssertHostMemory(stacked_data);
            solver
                .set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_point_data_from_stacked_host(
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
          },
          pybind11::arg("stacked_data"),
          pybind11::arg("offset") = 0);
}

}  // namespace caspar