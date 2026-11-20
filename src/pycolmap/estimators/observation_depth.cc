// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

double* LogDepthScalePointer(py::array_t<double, py::array::c_style>& array) {
  if (array.ndim() != 1 || array.size() != 1)
    throw py::value_error("log_depth_scale must have shape (1,)");
  return array.mutable_data();
}

}  // namespace

void BindObservationDepth(py::module& m) {
  auto PyDepthResidualType =
      py::enum_<DepthResidualType>(m, "DepthResidualType")
          .value("LINEAR", DepthResidualType::kLinear)
          .value("LOG", DepthResidualType::kLog);
  AddStringToEnumConstructor(PyDepthResidualType);
  auto PyBADepthConstraintOptions =
      py::classh<BADepthConstraintOptions>(m, "BADepthConstraintOptions")
          .def(py::init<>())
          .def_readwrite("residual_type",
                         &BADepthConstraintOptions::residual_type);
  MakeDataclass(PyBADepthConstraintOptions);
  m.def(
      "append_ba_depth_constraints",
      [](CeresBundleAdjuster& bundle_adjuster,
         Reconstruction& reconstruction,
         image_t image_id,
         const FlatHashMap<point2D_t, std::pair<double, double>>& depths,
         py::array_t<double, py::array::c_style> log_depth_scale,
         const BADepthConstraintOptions& options,
         ceres::LossFunction* loss_function) {
        return AppendBADepthConstraints(*bundle_adjuster.Problem(),
                                        reconstruction,
                                        image_id,
                                        depths,
                                        LogDepthScalePointer(log_depth_scale),
                                        options,
                                        loss_function);
      },
      "bundle_adjuster"_a,
      "reconstruction"_a,
      "image_id"_a,
      "depths"_a,
      "log_depth_scale"_a.noconvert().none(false),
      "options"_a = BADepthConstraintOptions(),
      "loss_function"_a = py::none(),
      py::keep_alive<1, 2>(),
      py::keep_alive<1, 5>(),
      py::keep_alive<1, 7>(),
      "Append selected image depths with a caller-owned log scale. "
      "The loss function is shared by the selected observations.");
}
