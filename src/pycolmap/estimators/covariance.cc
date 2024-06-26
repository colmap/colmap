#include "colmap/estimators/covariance.h"

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
  m.def(
      "estimate_pose_covariance_from_ba_ceres_backend",
      [](ceres::Problem* problem,
         Reconstruction* reconstruction) -> py::object {
        std::map<image_t, Eigen::MatrixXd> image_id_to_covar;
        if (!EstimatePoseCovarianceCeresBackend(
                problem, reconstruction, image_id_to_covar))
          return py::none();
        return py::cast(image_id_to_covar);
      },
      py::arg("problem"),
      py::arg("reconstruction"));

  m.def(
      "estimate_pose_covariance_from_ba",
      [](ceres::Problem* problem,
         Reconstruction* reconstruction,
         double lambda) -> py::object {
        std::map<image_t, Eigen::MatrixXd> image_id_to_covar;
        if (!EstimatePoseCovariance(
                problem, reconstruction, image_id_to_covar, lambda))
          return py::none();
        return py::cast(image_id_to_covar);
      },
      py::arg("problem"),
      py::arg("reconstruction"),
      py::arg("lambda") = 1e-8);

  using EstimatorBase = BundleAdjustmentCovarianceEstimatorBase;
  py::class_<EstimatorBase>(m, "BundleAdjustmentCovarianceEstimatorBase")
      .def(
          "set_pose_blocks",
          [](EstimatorBase& self, std::vector<py::array_t<double>>& pyarrays) {
            std::vector<const double*> blocks =
                ConvertListOfPyArraysToConstPointers(pyarrays);
            return self.SetPoseBlocks(blocks);
          },
          py::arg("pose_blocks"))
      .def("has_block", &EstimatorBase::HasBlock, py::arg("parameter_block"))
      .def("has_pose_block",
           &EstimatorBase::HasPoseBlock,
           py::arg("parameter_block"))
      .def("has_reconstruction", &EstimatorBase::HasReconstruction)
      .def("has_pose", &EstimatorBase::HasPose, py::arg("image_id"))
      .def("compute", &EstimatorBase::Compute)
      .def("compute_full", &EstimatorBase::ComputeFull)
      .def("get_pose_covariance",
           py::overload_cast<>(&EstimatorBase::GetPoseCovariance, py::const_))
      .def("get_pose_covariance",
           py::overload_cast<image_t>(&EstimatorBase::GetPoseCovariance,
                                      py::const_),
           py::arg("image_id"))
      .def("get_pose_covariance",
           py::overload_cast<const std::vector<image_t>&>(
               &EstimatorBase::GetPoseCovariance, py::const_),
           py::arg("image_ids"))
      .def("get_pose_covariance",
           py::overload_cast<image_t, image_t>(
               &EstimatorBase::GetPoseCovariance, py::const_),
           py::arg("image_id1"),
           py::arg("image_id2"))
      .def(
          "get_pose_covariance",
          [](EstimatorBase& self, py::array_t<double>& pyarray) {
            py::buffer_info info = pyarray.request();
            return self.GetPoseCovariance((double*)info.ptr);
          },
          py::arg("paramter_block"))
      .def(
          "get_pose_covariance",
          [](EstimatorBase& self, std::vector<py::array_t<double>>& pyarrays) {
            std::vector<double*> blocks;
            for (auto it = pyarrays.begin(); it != pyarrays.end(); ++it) {
              py::buffer_info info = it->request();
              blocks.push_back((double*)info.ptr);
            }
            return self.GetPoseCovariance(blocks);
          },
          py::arg("parameter_blocks"))
      .def(
          "get_pose_covariance",
          [](EstimatorBase& self,
             py::array_t<double>& pyarray1,
             py::array_t<double>& pyarray2) {
            py::buffer_info info1 = pyarray1.request();
            py::buffer_info info2 = pyarray2.request();
            return self.GetPoseCovariance((double*)info1.ptr,
                                          (double*)info2.ptr);
          },
          py::arg("parameter_block1"),
          py::arg("parameter_block2"))

      .def(
          "get_covariance",
          [](EstimatorBase& self, py::array_t<double>& pyarray) {
            py::buffer_info info = pyarray.request();
            return self.GetCovariance((double*)info.ptr);
          },
          py::arg("paramter_block"))
      .def(
          "get_covariance",
          [](EstimatorBase& self, std::vector<py::array_t<double>>& pyarrays) {
            std::vector<double*> blocks;
            for (auto it = pyarrays.begin(); it != pyarrays.end(); ++it) {
              py::buffer_info info = it->request();
              blocks.push_back((double*)info.ptr);
            }
            return self.GetCovariance(blocks);
          },
          py::arg("parameter_blocks"))
      .def(
          "get_covariance",
          [](EstimatorBase& self,
             py::array_t<double>& pyarray1,
             py::array_t<double>& pyarray2) {
            py::buffer_info info1 = pyarray1.request();
            py::buffer_info info2 = pyarray2.request();
            return self.GetCovariance((double*)info1.ptr, (double*)info2.ptr);
          },
          py::arg("parameter_block1"),
          py::arg("parameter_block2"))
      .def("has_valid_pose_covariance", &EstimatorBase::HasValidPoseCovariance)
      .def("has_valid_full_covariance", &EstimatorBase::HasValidFullCovariance);

  py::class_<BundleAdjustmentCovarianceEstimatorCeresBackend, EstimatorBase>(
      m, "BundleAdjustmentCovarianceEstimatorCeresBackend")
      .def(py::init<ceres::Problem*, Reconstruction*>(),
           py::arg("problem"),
           py::arg("reconstruction"))
      .def(
          py::init([](ceres::Problem* problem,
                      std::vector<py::array_t<double>>& pose_blocks_pyarrays,
                      std::vector<py::array_t<double>>& point_blocks_pyarrays) {
            std::vector<const double*> pose_blocks =
                ConvertListOfPyArraysToConstPointers(pose_blocks_pyarrays);
            std::vector<const double*> point_blocks =
                ConvertListOfPyArraysToConstPointers(point_blocks_pyarrays);
            return new BundleAdjustmentCovarianceEstimatorCeresBackend(
                problem, pose_blocks, point_blocks);
          }),
          py::arg("problem"),
          py::arg("pose_blocks"),
          py::arg("point_blocks"));

  py::class_<BundleAdjustmentCovarianceEstimator, EstimatorBase>(
      m, "BundleAdjustmentCovarianceEstimator")
      .def(py::init<ceres::Problem*, Reconstruction*, double>(),
           py::arg("problem"),
           py::arg("reconstruction"),
           py::arg("lambda") = 1e-8)
      .def(py::init([](ceres::Problem* problem,
                       std::vector<py::array_t<double>>& pose_blocks_pyarrays,
                       std::vector<py::array_t<double>>& point_blocks_pyarrays,
                       const double lambda) {
             std::vector<const double*> pose_blocks =
                 ConvertListOfPyArraysToConstPointers(pose_blocks_pyarrays);
             std::vector<const double*> point_blocks =
                 ConvertListOfPyArraysToConstPointers(point_blocks_pyarrays);
             return new BundleAdjustmentCovarianceEstimator(
                 problem, pose_blocks, point_blocks, lambda);
           }),
           py::arg("problem"),
           py::arg("pose_blocks"),
           py::arg("point_blocks"),
           py::arg("lambda") = 1e-8);
}
