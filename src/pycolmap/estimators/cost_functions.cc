#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/rigid3.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCostFunctions(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("cost_functions");
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  auto PyCovarianceType = py::enum_<CovarianceType>(m, "CovarianceType")
                              .value("IDENTITY", CovarianceType::IDENTITY)
                              .value("DIAGONAL", CovarianceType::DIAGONAL)
                              .value("DENSE", CovarianceType::DENSE);
  AddStringToEnumConstructor(PyCovarianceType);

  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorCostFunction,
                            const Eigen::Vector2d&,
                            const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point2D_covar"_a =
            Eigen::Matrix2d::Identity(),  // Useless variable. This must be
                                          // identity.
        "Reprojection error.");
  m.def("WeightedReprojErrorCost",
        &WeightedCameraCostFunction<ReprojErrorCostFunction,
                                    const Eigen::Vector2d&,
                                    const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "covariance_type"_a,
        "point2D"_a,
        "point2D_covar"_a = Eigen::Matrix2d::Identity(),
        "Reprojection error with 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoseCostFunction,
                            const Rigid3d&,
                            const Eigen::Vector2d&,
                            const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "point2D_covar"_a =
            Eigen::Matrix2d::Identity(),  // Useless variable. This must be
                                          // identity.
        "Reprojection error with constant camera pose.");
  m.def("WeightedReprojErrorCost",
        &WeightedCameraCostFunction<ReprojErrorConstantPoseCostFunction,
                                    const Rigid3d&,
                                    const Eigen::Vector2d&,
                                    const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "covariance_type"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "point2D_covar"_a = Eigen::Matrix2d::Identity(),
        "Reprojection error with constant camera pose and 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoint3DCostFunction,
                            const Eigen::Vector2d&,
                            const Eigen::Vector3d&,
                            const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point3D"_a,
        "point2D_covar"_a =
            Eigen::Matrix2d::Identity(),  // Useless variable. This must be
                                          // identity.
        "Reprojection error with constant 3D point.");
  m.def("WeightedReprojErrorCost",
        &WeightedCameraCostFunction<ReprojErrorConstantPoint3DCostFunction,
                                    const Eigen::Vector2d&,
                                    const Eigen::Vector3d&,
                                    const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "covariance_type"_a,
        "point2D"_a,
        "point3D"_a,
        "point2D_covar"_a = Eigen::Matrix2d::Identity(),
        "Reprojection error with constant 3D point and 2D detection noise.");

  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorCostFunction,
                            const Eigen::Vector2d&,
                            const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point2D_covar"_a =
            Eigen::Matrix2d::Identity(),  // Useless variable. This must be
                                          // identity.
        "Reprojection error for camera rig.");
  m.def("WeightedRigReprojErrorCost",
        &WeightedCameraCostFunction<RigReprojErrorCostFunction,
                                    const Eigen::Vector2d&,
                                    const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "covariance_type"_a,
        "point2D"_a,
        "point2D_covar"_a = Eigen::Matrix2d::Identity(),
        "Reprojection error for camera rig with 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorConstantRigCostFunction,
                            const Rigid3d&,
                            const Eigen::Vector2d&,
                            const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "point2D_covar"_a =
            Eigen::Matrix2d::Identity(),  // Useless variable. This must be
                                          // identity.
        "Reprojection error for camera rig with constant cam-from-rig pose.");
  m.def("WeightedRigReprojErrorCost",
        &WeightedCameraCostFunction<RigReprojErrorConstantRigCostFunction,
                                    const Rigid3d&,
                                    const Eigen::Vector2d&,
                                    const Eigen::Matrix2d&>,
        "camera_model_id"_a,
        "covariance_type"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "point2D_covar"_a = Eigen::Matrix2d::Identity(),
        "Reprojection error for camera rig with constant cam-from-rig pose and "
        "2D detection noise.");

  m.def("SampsonErrorCost",
        &SampsonErrorCostFunction::Create,
        "point2D1"_a,
        "point2D2"_a,
        "Sampson error for two-view geometry.");

  m.def("AbsolutePoseErrorCost",
        static_cast<ceres::CostFunction* (*)(const Rigid3d&,
                                             const EigenMatrix6d&)>(
            AbsolutePoseErrorCostFunction<CovarianceType::DENSE>::Create),
        "cam_from_world"_a,
        "covariance_cam"_a,
        "6-DoF error on the absolute pose.");
  m.def("MetricRelativePoseErrorCost",
        static_cast<ceres::CostFunction* (*)(const Rigid3d&,
                                             const EigenMatrix6d&)>(
            MetricRelativePoseErrorCostFunction<CovarianceType::DENSE>::Create),
        "i_from_j"_a,
        "covariance_j"_a,
        "6-DoF error between two absolute poses based on their relative pose.");
  m.def("Point3dAlignmentCost",
        static_cast<ceres::CostFunction* (*)(const Eigen::Vector3d&,
                                             const Eigen::Matrix3d&)>(
            Point3dAlignmentCostFunction<CovarianceType::DENSE>::Create),
        "ref_point"_a,
        "covariance_point"_a,
        "Error between 3D points transformed by a similarity transform.");
}
