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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                solver.SetPointNodesFromStackedHost(AsDoublePtr(stacked_data),
                                                    offset,
                                                    GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "set_Point_nodes_from_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.SetPointNodesFromStackedDevice(AsDoublePtr(stacked_data),
                                                      offset,
                                                      GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_Point_nodes_to_stacked_host",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertHostMemory(stacked_data);
                solver.GetPointNodesToStackedHost(AsDoublePtr(stacked_data),
                                                  offset,
                                                  GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_nodes"),
              pybind11::arg("offset") = 0)
          .def(
              "get_Point_nodes_to_stacked_device",
              [](GraphSolver& solver,
                 pybind11::object stacked_data,
                 size_t offset) {
                AssertDeviceMemory(stacked_data);
                solver.GetPointNodesToStackedDevice(AsDoublePtr(stacked_data),
                                                    offset,
                                                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                    AsDoublePtr(stacked_data),
                    offset,
                    GetNumRows(stacked_data));
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
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
                        AsDoublePtr(stacked_data),
                        offset,
                        GetNumRows(stacked_data));
              },
              pybind11::arg("stacked_data"),
              pybind11::arg("offset") = 0);
}

}  // namespace caspar