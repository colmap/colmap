// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_ceres.h"

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCeresRotationAverager(py::module& m) {
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  auto PyOptions =
      py::classh<CeresRotationAveragerOptions>(m,
                                               "CeresRotationAveragerOptions")
          .def(py::init<>())
          .def_readwrite("loss_function_type",
                         &CeresRotationAveragerOptions::loss_function_type,
                         "Loss function for relative-rotation residuals.")
          .def_readwrite("loss_function_scale",
                         &CeresRotationAveragerOptions::loss_function_scale,
                         "Loss function scale in radians.")
          .def_readwrite("reweighting",
                         &CeresRotationAveragerOptions::reweighting,
                         "Reweighting scheme for relative-rotation "
                         "constraints: UNIFORM or INLIER_MATCH_COUNT.")
          .def_readwrite("solver_options",
                         &CeresRotationAveragerOptions::solver_options,
                         "Ceres solver options.")
          .def_readwrite("skip_initialization",
                         &CeresRotationAveragerOptions::skip_initialization,
                         "Skip maximum spanning tree initialization.")
          .def_readwrite(
              "refine_sensor_from_rig",
              &CeresRotationAveragerOptions::refine_sensor_from_rig,
              "When False, treat each non-ref sensor's cam_from_rig rotation "
              "as a pre-calibrated constant.");
  MakeDataclass(PyOptions);

  py::classh<CeresRotationAverager>(m, "CeresRotationAverager")
      .def("solve",
           &CeresRotationAverager::Solve,
           py::call_guard<py::gil_scoped_release>(),
           "Optimize rotations in the reconstruction and return the solver "
           "summary.")
      .def("add_relative_rotation_residual",
           &CeresRotationAverager::AddRelativeRotationResidual,
           "image_id1"_a,
           "image_id2"_a,
           "cam2_from_cam1"_a,
           "loss"_a,
           "Add a relative-rotation residual between images whose frame and "
           "sensor rotations are configured in the problem.")
      .def_property_readonly(
          "problem",
          py::overload_cast<>(&CeresRotationAverager::Problem),
          py::return_value_policy::reference_internal,
          "Ceres problem for inspecting and modifying the optimization.")
      .def_readwrite("solver_options",
                     &CeresRotationAverager::solver_options,
                     "Ceres solver options.");

  m.def("create_default_ceres_rotation_averager",
        &CreateDefaultCeresRotationAverager,
        "options"_a,
        "pose_graph"_a,
        "reconstruction"_a,
        py::keep_alive<0, 3>(),
        "Create a Ceres rotation averaging problem that optimizes rotations "
        "directly in the reconstruction.");
}
