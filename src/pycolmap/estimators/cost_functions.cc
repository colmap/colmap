#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/rigid3.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename CameraModel>
using ReprojErrorCostFunctionWithNoise =
    IsotropicNoiseCostFunctionWrapper<ReprojErrorCostFunction<CameraModel>>;

template <typename CameraModel>
using ReprojErrorConstantPoseCostFunctionWithNoise =
    IsotropicNoiseCostFunctionWrapper<
        ReprojErrorConstantPoseCostFunction<CameraModel>>;

template <typename CameraModel>
using ReprojErrorConstantPoint3DCostFunctionWithNoise =
    IsotropicNoiseCostFunctionWrapper<
        ReprojErrorConstantPoint3DCostFunction<CameraModel>>;

template <typename CameraModel>
using RigReprojErrorCostFunctionWithNoise =
    IsotropicNoiseCostFunctionWrapper<RigReprojErrorCostFunction<CameraModel>>;

template <typename CameraModel>
using RigReprojErrorConstantRigCostFunctionWithNoise =
    IsotropicNoiseCostFunctionWrapper<
        RigReprojErrorConstantRigCostFunction<CameraModel>>;

void BindCostFunctions(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("cost_functions");
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorCostFunction, const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorCostFunctionWithNoise,
                            const double,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error with 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoseCostFunction,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoseCostFunctionWithNoise,
                            const double,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose and 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoint3DCostFunction,
                            const Eigen::Vector2d&,
                            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoint3DCostFunctionWithNoise,
                            const double,
                            const Eigen::Vector2d&,
                            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point and 2D detection noise.");

  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorCostFunction, const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error for camera rig.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorCostFunctionWithNoise,
                            const double,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error for camera rig with 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorConstantRigCostFunction,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorConstantRigCostFunctionWithNoise,
                            const double,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose and "
        "2D detection noise.");

  m.def("SampsonErrorCost",
        &SampsonErrorCostFunction::Create,
        "point2D1"_a,
        "point2D2"_a,
        "Sampson error for two-view geometry.");

  m.def("AbsolutePoseErrorCost",
        &AbsolutePoseErrorCostFunction::Create,
        "cam_from_world"_a,
        "covariance_cam"_a,
        "6-DoF error on the absolute pose.");
  m.def("MetricRelativePoseErrorCost",
        &MetricRelativePoseErrorCostFunction::Create,
        "i_from_j"_a,
        "covariance_j"_a,
        "6-DoF error between two absolute poses based on their relative pose.");
  m.def("Point3dAlignmentCost",
        &Point3dAlignmentCostFunction::Create,
        "ref_point"_a,
        "covariance_point"_a,
        "Error between 3D points transformed by a similarity transform.");
}
