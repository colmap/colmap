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
  py::
      class_<GraphSolver>(
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
              py::arg("simple_radial_merged_fixed_pose_fixed_point_num_max") =
                  0,
              py::arg("pinhole_merged_num_max") = 0,
              py::arg("pinhole_merged_fixed_pose_num_max") = 0,
              py::arg("pinhole_merged_fixed_point_num_max") = 0,
              py::arg("pinhole_merged_fixed_pose_fixed_point_num_max") = 0,
              py::arg("simple_radial_fixed_focal_and_extra_num_max") = 0,
              py::arg("simple_radial_fixed_principal_point_num_max") = 0,
              py::arg(
                  "simple_radial_fixed_pose_fixed_focal_and_extra_num_max") = 0,
              py::arg(
                  "simple_radial_fixed_pose_fixed_principal_point_num_max") = 0,
              py::arg("simple_radial_fixed_focal_and_extra_fixed_principal_"
                      "point_num_max") = 0,
              py::arg(
                  "simple_radial_fixed_focal_and_extra_fixed_point_num_max") =
                  0,
              py::arg(
                  "simple_radial_fixed_principal_point_fixed_point_num_max") =
                  0,
              py::arg("simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
                      "principal_point_num_max") = 0,
              py::arg("simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
                      "point_num_max") = 0,
              py::arg("simple_radial_fixed_pose_fixed_principal_point_fixed_"
                      "point_num_max") = 0,
              py::arg("simple_radial_fixed_focal_and_extra_fixed_principal_"
                      "point_fixed_point_num_max") = 0,
              py::arg("pinhole_fixed_focal_and_extra_num_max") = 0,
              py::arg("pinhole_fixed_principal_point_num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_focal_and_extra_num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_principal_point_num_max") = 0,
              py::arg("pinhole_fixed_focal_and_extra_fixed_principal_point_num_"
                      "max") = 0,
              py::arg("pinhole_fixed_focal_and_extra_fixed_point_num_max") = 0,
              py::arg("pinhole_fixed_principal_point_fixed_point_num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_focal_and_extra_fixed_"
                      "principal_point_num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_"
                      "num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_principal_point_fixed_point_"
                      "num_max") = 0,
              py::arg("pinhole_fixed_focal_and_extra_fixed_principal_point_"
                      "fixed_point_num_max") = 0)

          .def("set_params", &GraphSolver::set_params)
          .def("solve",
               &GraphSolver::solve,
               py::call_guard<py::gil_scoped_release>(),
               py::arg("print_progress") = false,
               py::arg("verbose_logging") = false)
          .def("finish_indices", &GraphSolver::finish_indices)
          .def("get_allocation_size", &GraphSolver::get_allocation_size)

          .def("set_PinholeCalib_num", &GraphSolver::SetPinholeCalibNum)
          .def(
              "set_PinholeCalib_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeCalibNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_PinholeCalib_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeCalibNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeCalib_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPinholeCalibNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeCalib_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPinholeCalibNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_PinholeFocalAndExtra_num",
               &GraphSolver::SetPinholeFocalAndExtraNum)
          .def(
              "set_PinholeFocalAndExtra_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFocalAndExtraNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_PinholeFocalAndExtra_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFocalAndExtraNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeFocalAndExtra_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPinholeFocalAndExtraNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeFocalAndExtra_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPinholeFocalAndExtraNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_PinholePose_num", &GraphSolver::SetPinholePoseNum)
          .def(
              "set_PinholePose_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholePoseNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_PinholePose_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholePoseNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholePose_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPinholePoseNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholePose_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPinholePoseNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_PinholePrincipalPoint_num",
               &GraphSolver::SetPinholePrincipalPointNum)
          .def(
              "set_PinholePrincipalPoint_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholePrincipalPointNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_PinholePrincipalPoint_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholePrincipalPointNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholePrincipalPoint_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPinholePrincipalPointNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholePrincipalPoint_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPinholePrincipalPointNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_Point_num", &GraphSolver::SetPointNum)
          .def(
              "set_Point_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPointNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_Point_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPointNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_Point_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPointNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_Point_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPointNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_SimpleRadialCalib_num",
               &GraphSolver::SetSimpleRadialCalibNum)
          .def(
              "set_SimpleRadialCalib_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialCalibNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_SimpleRadialCalib_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialCalibNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialCalib_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetSimpleRadialCalibNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialCalib_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetSimpleRadialCalibNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_SimpleRadialFocalAndExtra_num",
               &GraphSolver::SetSimpleRadialFocalAndExtraNum)
          .def(
              "set_SimpleRadialFocalAndExtra_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFocalAndExtraNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_SimpleRadialFocalAndExtra_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFocalAndExtraNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialFocalAndExtra_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetSimpleRadialFocalAndExtraNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialFocalAndExtra_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetSimpleRadialFocalAndExtraNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_SimpleRadialPose_num", &GraphSolver::SetSimpleRadialPoseNum)
          .def(
              "set_SimpleRadialPose_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialPoseNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_SimpleRadialPose_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialPoseNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialPose_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetSimpleRadialPoseNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialPose_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetSimpleRadialPoseNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_SimpleRadialPrincipalPoint_num",
               &GraphSolver::SetSimpleRadialPrincipalPointNum)
          .def(
              "set_SimpleRadialPrincipalPoint_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialPrincipalPointNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_SimpleRadialPrincipalPoint_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialPrincipalPointNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialPrincipalPoint_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetSimpleRadialPrincipalPointNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialPrincipalPoint_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetSimpleRadialPrincipalPointNodesToStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)

          .def("set_simple_radial_merged_num",
               &GraphSolver::SetSimpleRadialMergedNum)

          .def("set_simple_radial_merged_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_merged_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialMergedPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialMergedPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_merged_fixed_pose_num",
               &GraphSolver::SetSimpleRadialMergedFixedPoseNum)

          .def("set_simple_radial_merged_fixed_pose_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedFixedPoseCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_pose_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedFixedPoseCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_pose_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedFixedPosePointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_pose_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedFixedPosePointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPosePixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_pose_pixel_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPosePixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_pose_pose_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPosePoseDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_pose_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPosePoseDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_merged_fixed_point_num",
               &GraphSolver::SetSimpleRadialMergedFixedPointNum)

          .def("set_simple_radial_merged_fixed_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedFixedPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_point_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedFixedPointPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_point_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialMergedFixedPointCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_merged_fixed_point_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialMergedFixedPointCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_merged_fixed_point_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialMergedFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_point_pixel_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_point_point_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialMergedFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_merged_fixed_point_point_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialMergedFixedPointPointDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_merged_fixed_pose_fixed_point_num",
               &GraphSolver::SetSimpleRadialMergedFixedPoseFixedPointNum)

          .def(
              "set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_"
              "from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_merged_fixed_pose_fixed_point_calib_indices_"
              "from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialMergedFixedPoseFixedPointCalibIndicesFromDevice(
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialMergedFixedPoseFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_merged_num", &GraphSolver::SetPinholeMergedNum)

          .def("set_pinhole_merged_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_merged_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeMergedPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeMergedPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_merged_fixed_pose_num",
               &GraphSolver::SetPinholeMergedFixedPoseNum)

          .def("set_pinhole_merged_fixed_pose_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedFixedPoseCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_pose_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedFixedPoseCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_pose_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedFixedPosePointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_pose_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedFixedPosePointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_merged_fixed_pose_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeMergedFixedPosePixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeMergedFixedPosePixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeMergedFixedPosePoseDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeMergedFixedPosePoseDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_merged_fixed_point_num",
               &GraphSolver::SetPinholeMergedFixedPointNum)

          .def("set_pinhole_merged_fixed_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedFixedPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_point_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedFixedPointPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_point_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeMergedFixedPointCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_merged_fixed_point_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeMergedFixedPointCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_merged_fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeMergedFixedPointPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeMergedFixedPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeMergedFixedPointPointDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeMergedFixedPointPointDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_merged_fixed_pose_fixed_point_num",
               &GraphSolver::SetPinholeMergedFixedPoseFixedPointNum)

          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeMergedFixedPoseFixedPointCalibIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_calib_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeMergedFixedPoseFixedPointCalibIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_pixel_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_pose_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_point_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_merged_fixed_pose_fixed_point_point_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeMergedFixedPoseFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_focal_and_extra_num",
               &GraphSolver::SetSimpleRadialFixedFocalAndExtraNum)

          .def("set_simple_radial_fixed_focal_and_extra_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedFocalAndExtraPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_fixed_focal_and_extra_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetSimpleRadialFixedFocalAndExtraPoseIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_principal_point_indices_"
              "from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_principal_point_indices_"
              "from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetSimpleRadialFixedFocalAndExtraPointIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetSimpleRadialFixedFocalAndExtraPointIndicesFromDevice(
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
                    .SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedFocalAndExtraPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_focal_and_extra_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_principal_point_num",
               &GraphSolver::SetSimpleRadialFixedPrincipalPointNum)

          .def("set_simple_radial_fixed_principal_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedPrincipalPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_fixed_principal_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetSimpleRadialFixedPrincipalPointPoseIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_focal_and_extra_indices_"
              "from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_focal_and_extra_indices_"
              "from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetSimpleRadialFixedPrincipalPointPointIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetSimpleRadialFixedPrincipalPointPointIndicesFromDevice(
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
                    .SetSimpleRadialFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_principal_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_principal_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_pose_fixed_focal_and_extra_num",
               &GraphSolver::SetSimpleRadialFixedPoseFixedFocalAndExtraNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_principal_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_pose_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_"
              "extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_focal_and_"
              "extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_pose_fixed_principal_point_num",
               &GraphSolver::SetSimpleRadialFixedPoseFixedPrincipalPointNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_focal_and_"
              "extra_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_focal_and_"
              "extra_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_pose_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_pose_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_principal_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_principal_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "num",
              &GraphSolver::
                  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointNum)

          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "focal_and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "focal_and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_focal_and_extra_fixed_point_num",
               &GraphSolver::SetSimpleRadialFixedFocalAndExtraFixedPointNum)

          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_principal_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_principal_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_"
              "extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_focal_and_"
              "extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_point_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_principal_point_fixed_point_num",
               &GraphSolver::SetSimpleRadialFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_focal_and_"
              "extra_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_focal_and_"
              "extra_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_principal_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_principal_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_principal_point_fixed_point_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_num",
              &GraphSolver::
                  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_focal_and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_focal_and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_"
              "principal_point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "num",
              &GraphSolver::
                  SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "principal_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
              "num",
              &GraphSolver::
                  SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
              "focal_and_extra_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
              "focal_and_extra_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
              "pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_principal_point_fixed_point_"
              "pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetSimpleRadialFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_num",
              &GraphSolver::
                  SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_focal_and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_focal_and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_focal_and_extra_fixed_principal_point_"
              "fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_focal_and_extra_num",
               &GraphSolver::SetPinholeFixedFocalAndExtraNum)

          .def("set_pinhole_fixed_focal_and_extra_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedFocalAndExtraPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_focal_and_extra_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedFocalAndExtraPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_focal_and_extra_principal_point_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_principal_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def("set_pinhole_fixed_focal_and_extra_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedFocalAndExtraPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_focal_and_extra_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedFocalAndExtraPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedFocalAndExtraPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedFocalAndExtraPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_focal_and_extra_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_principal_point_num",
               &GraphSolver::SetPinholeFixedPrincipalPointNum)

          .def("set_pinhole_fixed_principal_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPrincipalPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_principal_point_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPrincipalPointPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_principal_point_focal_and_extra_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_principal_point_focal_and_extra_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def("set_pinhole_fixed_principal_point_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPrincipalPointPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_principal_point_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPrincipalPointPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_principal_point_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPrincipalPointPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_principal_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPrincipalPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_principal_point_principal_point_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_principal_point_principal_point_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_pose_fixed_focal_and_extra_num",
               &GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraNum)

          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_principal_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPointIndicesFromDevice(
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
                    .SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedPoseFixedFocalAndExtraPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_pose_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_focal_and_extra_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_pose_fixed_principal_point_num",
               &GraphSolver::SetPinholeFixedPoseFixedPrincipalPointNum)

          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_focal_and_extra_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPointIndicesFromDevice(
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
                    .SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_pose_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_principal_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_principal_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_focal_and_extra_fixed_principal_point_num",
               &GraphSolver::SetPinholeFixedFocalAndExtraFixedPrincipalPointNum)

          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_pixel_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_"
              "and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_focal_"
              "and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_focal_and_extra_fixed_point_num",
               &GraphSolver::SetPinholeFixedFocalAndExtraFixedPointNum)

          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_principal_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
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
                    .SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_point_focal_and_extra_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedFocalAndExtraFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_principal_point_fixed_point_num",
               &GraphSolver::SetPinholeFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_principal_point_fixed_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_principal_point_fixed_point_focal_and_extra_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
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
                    .SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_principal_point_fixed_point_principal_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_principal_point_fixed_point_principal_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
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
                    .SetPinholeFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_num",
              &GraphSolver::
                  SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointNum)

          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_focal_and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_focal_and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_principal_"
              "point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_num",
               &GraphSolver::SetPinholeFixedPoseFixedFocalAndExtraFixedPointNum)

          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_"
              "principal_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pixel_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_pose_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_"
              "and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_focal_"
              "and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedFocalAndExtraFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_num",
              &GraphSolver::SetPinholeFixedPoseFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_"
              "and_extra_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_focal_"
              "and_extra_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointFocalAndExtraIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pixel_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_pose_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_principal_point_fixed_point_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_num",
              &GraphSolver::
                  SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_focal_and_extra_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_focal_and_extra_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointFocalAndExtraDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_"
              "point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeFixedFocalAndExtraFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0);
}

}  // namespace caspar