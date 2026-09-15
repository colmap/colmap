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
  auto PyOptions =
      py::classh<CeresRotationAveragerOptions>(m,
                                               "CeresRotationAveragerOptions")
          .def(py::init<>())
          .def_readwrite("loss_function_type",
                         &CeresRotationAveragerOptions::loss_function_type)
          .def_readwrite("loss_function_scale",
                         &CeresRotationAveragerOptions::loss_function_scale)
          .def_readwrite("solver_options",
                         &CeresRotationAveragerOptions::solver_options)
          .def_readwrite("skip_initialization",
                         &CeresRotationAveragerOptions::skip_initialization);
  MakeDataclass(PyOptions);

  py::classh<CeresRotationAverager>(m, "CeresRotationAverager")
      .def("solve", &CeresRotationAverager::Solve)
      .def("add_relative_rotation_residual",
           &CeresRotationAverager::AddRelativeRotationResidual,
           "image_id1"_a,
           "image_id2"_a,
           "cam2_from_cam1"_a,
           "loss"_a)
      .def_property_readonly(
          "problem",
          py::overload_cast<>(&CeresRotationAverager::Problem),
          py::return_value_policy::reference_internal)
      .def_property_readonly("solver_options",
                             &CeresRotationAverager::SolverOptions,
                             py::return_value_policy::copy);

  m.def("create_default_ceres_rotation_averager",
        &CreateDefaultCeresRotationAverager,
        "options"_a,
        "pose_graph"_a,
        "reconstruction"_a,
        py::keep_alive<0, 3>());
}
