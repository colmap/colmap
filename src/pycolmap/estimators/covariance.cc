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
      .def_property(
          "qvec",
          [](internal::PoseParam& self)
              -> py::typing::Optional<py::array_t<double>> {
            if (!self.qvec)
              return py::none();
            else
              return py::array_t<double>(4, self.qvec);
          },
          [](internal::PoseParam& self, py::array_t<double> pyarray) {
            THROW_CHECK_EQ(pyarray.ndim(), 1);
            THROW_CHECK_EQ(pyarray.size(), 4);
            py::buffer_info info = pyarray.request();
            self.qvec = (double*)info.ptr;
          })
      .def_property(
          "tvec",
          [](internal::PoseParam& self)
              -> py::typing::Optional<py::array_t<double>> {
            if (!self.tvec)
              return py::none();
            else
              return py::array_t<double>(3, self.tvec);
          },
          [](internal::PoseParam& self, py::array_t<double> pyarray) {
            THROW_CHECK_EQ(pyarray.ndim(), 1);
            THROW_CHECK_EQ(pyarray.size(), 3);
            py::buffer_info info = pyarray.request();
            self.tvec = (double*)info.ptr;
          });
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
           &BACovariance::GetPointCov,
           "image_id"_a,
           "Covariance for 3D points, conditioned on all other variables set "
           "constant. If some dimensions are kept constant, the respective "
           "rows/columns are omitted. Returns null if 3D point not a variable "
           "in the problem.")
      .def("get_cam_cov_from_world",
           &BACovariance::GetCamCovFromWorld,
           "image_id"_a,
           "Tangent space covariance in the order [rotation, translation]. If "
           "some dimensions are kept constant, the respective rows/columns are "
           "omitted. Returns null if image is not a variable in the problem.")
      .def("get_cam_cross_cov_from_world",
           &BACovariance::GetCamCrossCovFromWorld,
           "image_id1"_a,
           "image_id2"_a,
           "Tangent space covariance in the order [rotation, translation]. If "
           "some dimensions are kept constant, the respective rows/columns are "
           "omitted. Returns null if image is not a variable in the problem.")
      .def("get_cam2_cov_from_cam1",
           &BACovariance::GetCam2CovFromCam1,
           "image_id1"_a,
           "cam1_from_world"_a,
           "image_id2"_a,
           "cam2_from_world"_a,
           "Get relative pose covariance in the order [rotation, translation]. "
           "This function returns null if some dimensions are kept constant "
           "for either of the two poses. This does not mean that one cannot "
           "get relative pose covariance for such case, but requires custom "
           "logic to fill in zero block in the covariance matrix.")
      .def(
          "get_other_params_cov",
          [](BACovariance& self, py::array_t<double>& pyarray) {
            THROW_CHECK_EQ(pyarray.ndim(), 1);
            py::buffer_info info = pyarray.request();
            return self.GetOtherParamsCov((double*)info.ptr);
          },
          "param"_a,
          "Tangent space covariance for any variable parameter block in the "
          "problem. If some dimensions are kept constant, the respective "
          "rows/columns are omitted. Returns null if parameter block not a "
          "variable in the problem.");

  m.def(
      "estimate_ba_covariance_from_problem",
      &EstimateBACovarianceFromProblem,
      "options"_a,
      "reconstruction"_a,
      "problem"_a,
      "Computes covariances for the parameters in a bundle adjustment "
      "problem. It is important that the problem has a structure suitable for "
      "solving using the Schur complement trick. This is the case for the "
      "standard configuration of bundle adjustment problems, but be careful "
      "if you modify the underlying problem with custom residuals. Returns "
      "null if the estimation was not successful.");

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
