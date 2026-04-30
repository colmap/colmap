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
              py::arg("PinholeFocal_num_max") = 0,
              py::arg("PinholePose_num_max") = 0,
              py::arg("PinholePrincipalPoint_num_max") = 0,
              py::arg("Point_num_max") = 0,
              py::arg("SimpleRadialCalib_num_max") = 0,
              py::arg("SimpleRadialFocalAndDistortion_num_max") = 0,
              py::arg("SimpleRadialPose_num_max") = 0,
              py::arg("SimpleRadialPrincipalPoint_num_max") = 0,
              py::arg("simple_radial_num_max") = 0,
              py::arg("simple_radial_fixed_pose_num_max") = 0,
              py::arg("simple_radial_fixed_point_num_max") = 0,
              py::arg("simple_radial_fixed_pose_fixed_point_num_max") = 0,
              py::arg("pinhole_num_max") = 0,
              py::arg("pinhole_fixed_pose_num_max") = 0,
              py::arg("pinhole_fixed_point_num_max") = 0,
              py::arg("pinhole_fixed_pose_fixed_point_num_max") = 0,
              py::arg(
                  "simple_radial_split_fixed_focal_and_distortion_num_max") = 0,
              py::arg("simple_radial_split_fixed_principal_point_num_max") = 0,
              py::arg("simple_radial_split_fixed_pose_fixed_focal_and_"
                      "distortion_num_max") = 0,
              py::arg("simple_radial_split_fixed_pose_fixed_principal_point_"
                      "num_max") = 0,
              py::arg("simple_radial_split_fixed_focal_and_distortion_fixed_"
                      "principal_point_num_max") = 0,
              py::arg("simple_radial_split_fixed_focal_and_distortion_fixed_"
                      "point_num_max") = 0,
              py::arg("simple_radial_split_fixed_principal_point_fixed_point_"
                      "num_max") = 0,
              py::arg("simple_radial_split_fixed_pose_fixed_focal_and_"
                      "distortion_fixed_principal_point_num_max") = 0,
              py::arg("simple_radial_split_fixed_pose_fixed_focal_and_"
                      "distortion_fixed_point_num_max") = 0,
              py::arg("simple_radial_split_fixed_pose_fixed_principal_point_"
                      "fixed_point_num_max") = 0,
              py::arg("simple_radial_split_fixed_focal_and_distortion_fixed_"
                      "principal_point_fixed_point_num_max") = 0,
              py::arg("pinhole_split_fixed_focal_num_max") = 0,
              py::arg("pinhole_split_fixed_principal_point_num_max") = 0,
              py::arg("pinhole_split_fixed_pose_fixed_focal_num_max") = 0,
              py::arg(
                  "pinhole_split_fixed_pose_fixed_principal_point_num_max") = 0,
              py::arg(
                  "pinhole_split_fixed_focal_fixed_principal_point_num_max") =
                  0,
              py::arg("pinhole_split_fixed_focal_fixed_point_num_max") = 0,
              py::arg(
                  "pinhole_split_fixed_principal_point_fixed_point_num_max") =
                  0,
              py::arg("pinhole_split_fixed_pose_fixed_focal_fixed_principal_"
                      "point_num_max") = 0,
              py::arg(
                  "pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max") =
                  0,
              py::arg("pinhole_split_fixed_pose_fixed_principal_point_fixed_"
                      "point_num_max") = 0,
              py::arg("pinhole_split_fixed_focal_fixed_principal_point_fixed_"
                      "point_num_max") = 0)

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

          .def("set_PinholeFocal_num", &GraphSolver::SetPinholeFocalNum)
          .def(
              "set_PinholeFocal_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFocalNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_PinholeFocal_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFocalNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeFocal_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPinholeFocalNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_PinholeFocal_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPinholeFocalNodesToStackedDevice(
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

          .def("set_SimpleRadialFocalAndDistortion_num",
               &GraphSolver::SetSimpleRadialFocalAndDistortionNum)
          .def(
              "set_SimpleRadialFocalAndDistortion_nodes_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFocalAndDistortionNodesFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_SimpleRadialFocalAndDistortion_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFocalAndDistortionNodesFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialFocalAndDistortion_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetSimpleRadialFocalAndDistortionNodesToStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_SimpleRadialFocalAndDistortion_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetSimpleRadialFocalAndDistortionNodesToStackedDevice(
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

          .def("set_simple_radial_num", &GraphSolver::SetSimpleRadialNum)

          .def("set_simple_radial_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialPoseIndicesFromHost(AsUintPtr(indices),
                                                           GetNumRows(indices));
               })
          .def("set_simple_radial_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_pose_num",
               &GraphSolver::SetSimpleRadialFixedPoseNum)

          .def("set_simple_radial_fixed_pose_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedPoseCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_pose_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialFixedPoseCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_pose_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedPosePointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_pose_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialFixedPosePointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_fixed_pose_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFixedPosePixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFixedPosePixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFixedPosePoseDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFixedPosePoseDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_point_num",
               &GraphSolver::SetSimpleRadialFixedPointNum)

          .def("set_simple_radial_fixed_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_point_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialFixedPointPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_point_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetSimpleRadialFixedPointCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_simple_radial_fixed_point_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetSimpleRadialFixedPointCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_simple_radial_fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFixedPointPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFixedPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetSimpleRadialFixedPointPointDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetSimpleRadialFixedPointPointDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_fixed_pose_fixed_point_num",
               &GraphSolver::SetSimpleRadialFixedPoseFixedPointNum)

          .def(
              "set_simple_radial_fixed_pose_fixed_point_calib_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetSimpleRadialFixedPoseFixedPointCalibIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_point_calib_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetSimpleRadialFixedPoseFixedPointCalibIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_fixed_pose_fixed_point_pixel_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_point_pixel_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_point_pose_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_point_point_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_fixed_pose_fixed_point_point_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_num", &GraphSolver::SetPinholeNum)

          .def("set_pinhole_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholePoseIndicesFromHost(AsUintPtr(indices),
                                                      GetNumRows(indices));
               })
          .def("set_pinhole_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholePoseIndicesFromDevice(AsUintPtr(indices),
                                                        GetNumRows(indices));
               })
          .def("set_pinhole_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeCalibIndicesFromHost(AsUintPtr(indices),
                                                       GetNumRows(indices));
               })
          .def("set_pinhole_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeCalibIndicesFromDevice(AsUintPtr(indices),
                                                         GetNumRows(indices));
               })
          .def("set_pinhole_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholePointIndicesFromHost(AsUintPtr(indices),
                                                       GetNumRows(indices));
               })
          .def("set_pinhole_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholePointIndicesFromDevice(AsUintPtr(indices),
                                                         GetNumRows(indices));
               })
          .def(
              "set_pinhole_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholePixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholePixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_pose_num",
               &GraphSolver::SetPinholeFixedPoseNum)

          .def("set_pinhole_fixed_pose_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPoseCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_pose_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPoseCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_pose_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPosePointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_pose_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPosePointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_pose_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPosePixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPosePixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPosePoseDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPosePoseDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_point_num",
               &GraphSolver::SetPinholeFixedPointNum)

          .def("set_pinhole_fixed_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_point_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPointPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_point_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPointCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_point_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPointCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPointPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPointPointDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPointPointDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_fixed_pose_fixed_point_num",
               &GraphSolver::SetPinholeFixedPoseFixedPointNum)

          .def("set_pinhole_fixed_pose_fixed_point_calib_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_fixed_pose_fixed_point_calib_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeFixedPoseFixedPointCalibIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPoseDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPointDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_fixed_pose_fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_split_fixed_focal_and_distortion_num",
               &GraphSolver::SetSimpleRadialSplitFixedFocalAndDistortionNum)

          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_pose_indices_"
              "from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_pose_indices_"
              "from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_principal_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_principal_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_focal_and_"
              "distortion_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_focal_and_"
              "distortion_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_split_fixed_principal_point_num",
               &GraphSolver::SetSimpleRadialSplitFixedPrincipalPointNum)

          .def(
              "set_simple_radial_split_fixed_principal_point_pose_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_focal_and_"
              "distortion_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFocalAndDistortionIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_focal_and_"
              "distortion_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFocalAndDistortionIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_point_indices_"
              "from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_point_indices_"
              "from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_pixel_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_pixel_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_principal_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_principal_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionNum)

          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "principal_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "focal_and_distortion_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "focal_and_distortion_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_num",
              &GraphSolver::SetSimpleRadialSplitFixedPoseFixedPrincipalPointNum)

          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_focal_"
              "and_distortion_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndDistortionIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_focal_"
              "and_distortion_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndDistortionIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_pixel_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_pixel_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_pose_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_pose_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointNum)

          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_focal_and_distortion_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_focal_and_distortion_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointNum)

          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "principal_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "focal_and_distortion_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "focal_and_distortion_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_point_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_simple_radial_split_fixed_principal_point_fixed_point_num",
               &GraphSolver::
                   SetSimpleRadialSplitFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_focal_"
              "and_distortion_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_focal_"
              "and_distortion_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_pixel_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_pixel_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_point_"
              "data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_principal_point_fixed_point_point_"
              "data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointNum)

          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_focal_and_distortion_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_focal_and_distortion_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_principal_point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointNum)

          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_principal_point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_focal_and_distortion_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_focal_and_distortion_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_focal_and_distortion_"
              "fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_focal_and_distortion_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_focal_and_distortion_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_pose_fixed_principal_point_fixed_"
              "point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_num",
              &GraphSolver::
                  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointNum)

          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_focal_and_distortion_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointFocalAndDistortionDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_focal_and_distortion_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointFocalAndDistortionDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_principal_point_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_principal_point_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_simple_radial_split_fixed_focal_and_distortion_fixed_"
              "principal_point_fixed_point_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_focal_num",
               &GraphSolver::SetPinholeSplitFixedFocalNum)

          .def("set_pinhole_split_fixed_focal_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeSplitFixedFocalPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_split_fixed_focal_pose_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeSplitFixedFocalPoseIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_split_fixed_focal_principal_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeSplitFixedFocalPrincipalPointIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_principal_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedFocalPrincipalPointIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def("set_pinhole_split_fixed_focal_point_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeSplitFixedFocalPointIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def("set_pinhole_split_fixed_focal_point_indices_from_device",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertDeviceMemory(indices);
                 solver.SetPinholeSplitFixedFocalPointIndicesFromDevice(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_split_fixed_focal_pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeSplitFixedFocalPixelDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeSplitFixedFocalPixelDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_focal_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPinholeSplitFixedFocalFocalDataFromStackedDevice(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_focal_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.SetPinholeSplitFixedFocalFocalDataFromStackedHost(
                    AsFloatPtr(stacked_data), offset, GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_principal_point_num",
               &GraphSolver::SetPinholeSplitFixedPrincipalPointNum)

          .def("set_pinhole_split_fixed_principal_point_pose_indices_from_host",
               [](GraphSolver& solver, pybind11::object indices) {
                 AssertHostMemory(indices);
                 solver.SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
                     AsUintPtr(indices), GetNumRows(indices));
               })
          .def(
              "set_pinhole_split_fixed_principal_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedPrincipalPointPoseIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_focal_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_focal_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedPrincipalPointFocalIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedPrincipalPointPointIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_pixel_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_pixel_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_principal_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_principal_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_pose_fixed_focal_num",
               &GraphSolver::SetPinholeSplitFixedPoseFixedFocalNum)

          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_principal_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_principal_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_point_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeSplitFixedPoseFixedFocalPointIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_point_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedPoseFixedFocalPointIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_pixel_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_pixel_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_pose_data_from_stacked_"
              "device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_pose_data_from_stacked_"
              "host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_focal_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_focal_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_pose_fixed_principal_point_num",
               &GraphSolver::SetPinholeSplitFixedPoseFixedPrincipalPointNum)

          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_focal_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_focal_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_pose_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_pose_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_principal_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_principal_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_focal_fixed_principal_point_num",
               &GraphSolver::SetPinholeSplitFixedFocalFixedPrincipalPointNum)

          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_focal_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_focal_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_principal_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_principal_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_focal_fixed_point_num",
               &GraphSolver::SetPinholeSplitFixedFocalFixedPointNum)

          .def(
              "set_pinhole_split_fixed_focal_fixed_point_pose_indices_from_"
              "host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver.SetPinholeSplitFixedFocalFixedPointPoseIndicesFromHost(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_pose_indices_from_"
              "device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver.SetPinholeSplitFixedFocalFixedPointPoseIndicesFromDevice(
                    AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_principal_point_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_principal_point_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_pixel_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_pixel_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_focal_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_focal_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_point_data_from_"
              "stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_point_point_data_from_"
              "stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_principal_point_fixed_point_num",
               &GraphSolver::SetPinholeSplitFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_pose_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_pose_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_focal_"
              "indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_focal_"
              "indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_principal_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_principal_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_principal_point_fixed_point_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "num",
              &GraphSolver::
                  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum)

          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "focal_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "focal_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def("set_pinhole_split_fixed_pose_fixed_focal_fixed_point_num",
               &GraphSolver::SetPinholeSplitFixedPoseFixedFocalFixedPointNum)

          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_principal_"
              "point_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_principal_"
              "point_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pixel_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pixel_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pose_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_pose_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_focal_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_focal_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_point_data_"
              "from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_focal_fixed_point_point_data_"
              "from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "num",
              &GraphSolver::
                  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "focal_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "focal_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "pose_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "pose_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "num",
              &GraphSolver::
                  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum)

          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "pose_indices_from_host",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertHostMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "pose_indices_from_device",
              [](GraphSolver& solver, pybind11::object indices) {
                AssertDeviceMemory(indices);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromDevice(
                        AsUintPtr(indices), GetNumRows(indices));
              })
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "pixel_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "pixel_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "focal_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "focal_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "principal_point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "point_data_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0)
          .def(
              "set_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_"
              "point_data_from_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver
                    .SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
                        AsFloatPtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0);
}

}  // namespace caspar