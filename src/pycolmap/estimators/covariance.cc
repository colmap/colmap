#include "colmap/estimators/covariance.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

std::vector<const double*> ConvertListOfPyArraysToConstPointers(
    const std::vector<py::array_t<double>>& pyarrays) {
  std::vector<const double*> blocks;
  blocks.reserve(pyarrays.size());
  for (auto it = pyarrays.begin(); it != pyarrays.end(); ++it) {
    py::buffer_info info = it->request();
    blocks.push_back((const double*)info.ptr);
  }
  return blocks;
}

}  // namespace

void BindCovarianceEstimator(py::module& m) {
  auto PyBACovarianceOptionsParams =
      py::enum_<BACovarianceOptions::Params>(m, "BACovarianceOptionsParams")
          .value("POSES", BACovarianceOptions::Params::POSES)
          .value("POINTS", BACovarianceOptions::Params::POINTS)
          .value("POSES_AND_POINTS",
                 BACovarianceOptions::Params::POSES_AND_POINTS)
          .value("ALL", BACovarianceOptions::Params::ALL);
  AddStringToEnumConstructor(PyBACovarianceOptionsParams);

  py::class_<internal::PoseParam> PyExperimentalPoseParam(
      m, "ExperimentalPoseParam");
  PyExperimentalPoseParam.def(py::init<>())
      .def_readwrite("image_id", &internal::PoseParam::image_id)
      .def_readwrite("qvec", &internal::PoseParam::qvec)
      .def_readwrite("tvec", &internal::PoseParam::tvec);
  MakeDataclass(PyExperimentalPoseParam);

  py::class_<BACovarianceOptions> PyBACovarianceOptions(m,
                                                        "BACovarianceOptions");
  PyBACovarianceOptions.def(py::init<>())
      .def_readwrite("params",
                     &BACovarianceOptions::params,
                     "For which parameters to compute the covariance.")
      .def_readwrite(
          "damping",
          &BACovarianceOptions::damping,
          "Damping factor for the Hessian in the Schur complement solver. "
          "Enables to robustly deal with poorly conditioned parameters.")
      .def_readwrite(
          "experimental_custom_poses",
          &BACovarianceOptions::experimental_custom_poses,
          "WARNING: This option will be removed in a future release, use at "
          "your own risk. For custom bundle adjustment problems, this enables "
          "to specify a custom set of pose parameter blocks to consider. Note "
          "that these pose blocks must not necessarily be part of the "
          "reconstruction but they must follow the standard requirement for "
          "applying the Schur complement trick.");
  MakeDataclass(PyBACovarianceOptions);

  py::class_<BACovariance>(m, "BACovariance")
      .def("get_point_cov",
           &BACovariance::GetCamFromWorldCov,
           "image_id"_a,
           "Covariance for 3D points, conditioned on all other variables set "
           "constant. If some dimensions are kept constant, the respective "
           "rows/columns are omitted. Returns null if 3D point not a variable "
           "in the problem.")
      .def("get_cam_from_world_cov",
           &BACovariance::GetCamFromWorldCov,
           "image_id"_a,
           "Tangent space covariance in the order [rotation, translation]. If "
           "some dimensions are kept constant, the respective rows/columns are "
           "omitted. Returns null if image not a variable in the problem.")
      .def("get_cam1_from_cam2_cov",
           &BACovariance::GetCam1FromCam2Cov,
           "image_id1"_a,
           "image_id2"_a,
           "Tangent space covariance in the order [rotation, translation]. If "
           "some dimensions are kept constant, the respective rows/columns are "
           "omitted. Returns null if image not a variable in the problem.")
      .def("get_other_params_cov",
           &BACovariance::GetOtherParamsCov,
           "param"_a,
           "Tangent space covariance for any variable parameter block in the "
           "problem. If some dimensions are kept constant, the respective "
           "rows/columns are omitted. Returns null if parameter block not a "
           "variable in the problem.");

  m.def(
      "estimate_ba_covariance",
      &EstimateBACovariance,
      "options"_a,
      "reconstruction"_a,
      "bundle_adjuster"_a,
      "Computes covariances for the parameters in a bundle adjustment "
      "problem. It is important that the problem has a structure suitable for "
      "solving using the Schur complement trick. This is the case for the "
      "standard configuration of bundle adjustment problems, but be careful "
      "if you modify the underlying problem with custom residuals. Returns "
      "null if the estimation was not successful.");
}
