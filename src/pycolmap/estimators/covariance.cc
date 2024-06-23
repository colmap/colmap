#include "colmap/estimators/covariance.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCovarianceEstimator(py::module& m) {
  m.def(
      "estimate_pose_covariance_from_ba_ceres_backend",
      [](ceres::Problem* problem,
         Reconstruction* reconstruction) -> py::object {
        std::map<image_t, Eigen::MatrixXd> res;
        if (!EstimatePoseCovarianceCeresBackend(problem, reconstruction, res))
          return py::none();
        return py::cast(res);
      },
      py::arg("problem"),
      py::arg("reconstruction"));

  m.def(
      "estimate_pose_covariance_from_ba",
      [](ceres::Problem* problem,
         Reconstruction* reconstruction,
         double lambda) -> py::object {
        std::map<image_t, Eigen::MatrixXd> res;
        if (!EstimatePoseCovariance(problem, reconstruction, res, lambda))
          return py::none();
        return py::cast(res);
      },
      py::arg("problem"),
      py::arg("reconstruction"),
      py::arg("lambda") = 1e-6);

#define REGISTER_ESTIMATOR(NAME)                                               \
  .def(py::init<ceres::Problem*, Reconstruction*>())                           \
      .def("compute", &NAME::Compute)                                          \
      .def("compute_full", &NAME::ComputeFull)                                 \
      .def("get_pose_covariance",                                              \
           (Eigen::MatrixXd(NAME::*)() const) & NAME::GetPoseCovariance)       \
      .def(                                                                    \
          "get_pose_covariance",                                               \
          (Eigen::MatrixXd(NAME::*)(image_t) const) & NAME::GetPoseCovariance) \
      .def("get_pose_covariance",                                              \
           (Eigen::MatrixXd(NAME::*)(const std::vector<image_t>&) const) &     \
               NAME::GetPoseCovariance)                                        \
      .def("get_pose_covariance",                                              \
           (Eigen::MatrixXd(NAME::*)(image_t, image_t) const) &                \
               NAME::GetPoseCovariance)                                        \
      .def("get_covariance",                                                   \
           [](NAME& self, py::array_t<double>& pyarray) {                      \
             py::buffer_info info = pyarray.request();                         \
             return self.GetCovariance((double*)info.ptr);                     \
           })                                                                  \
      .def("get_covariance",                                                   \
           [](NAME& self, std::vector<py::array_t<double>>& pyarrays) {        \
             std::vector<double*> blocks;                                      \
             for (auto it = pyarrays.begin(); it != pyarrays.end(); ++it) {    \
               py::buffer_info info = it->request();                           \
               blocks.push_back((double*)info.ptr);                            \
             }                                                                 \
             return self.GetCovariance(blocks);                                \
           })                                                                  \
      .def("get_covariance",                                                   \
           [](NAME& self,                                                      \
              py::array_t<double>& pyarray1,                                   \
              py::array_t<double>& pyarray2) {                                 \
             py::buffer_info info1 = pyarray1.request();                       \
             py::buffer_info info2 = pyarray2.request();                       \
             return self.GetCovariance((double*)info1.ptr,                     \
                                       (double*)info2.ptr);                    \
           })                                                                  \
      .def("has_valid_pose_covariance", &NAME::HasValidPoseCovariance)         \
      .def("has_valid_full_covariance", &NAME::HasValidFullCovariance)

  py::class_<BundleAdjustmentCovarianceEstimatorCeresBackend>(
      m, "BundleAdjustmentCovarianceEstimatorCeresBackend")
      REGISTER_ESTIMATOR(BundleAdjustmentCovarianceEstimatorCeresBackend);

  py::class_<BundleAdjustmentCovarianceEstimator>(
      m, "BundleAdjustmentCovarianceEstimator")
      REGISTER_ESTIMATOR(BundleAdjustmentCovarianceEstimator)
          .def("set_lambda", &BundleAdjustmentCovarianceEstimator::SetLambda);

#undef REGISTER_ESTIMATOR
}
