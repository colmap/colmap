#include "colmap/estimators/cost_functions.h"

#include "colmap/geometry/rigid3.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename CameraModel>
using ReprojErrorCostFunctorWithNoise =
    IsotropicNoiseCostFunctorWrapper<ReprojErrorCostFunctor<CameraModel>>;

template <typename CameraModel>
using ReprojErrorConstantPoseCostFunctorWithNoise =
    IsotropicNoiseCostFunctorWrapper<
        ReprojErrorConstantPoseCostFunctor<CameraModel>>;

template <typename CameraModel>
using ReprojErrorConstantPoint3DCostFunctorWithNoise =
    IsotropicNoiseCostFunctorWrapper<
        ReprojErrorConstantPoint3DCostFunctor<CameraModel>>;

template <typename CameraModel>
using RigReprojErrorCostFunctorWithNoise =
    IsotropicNoiseCostFunctorWrapper<RigReprojErrorCostFunctor<CameraModel>>;

template <typename CameraModel>
using RigReprojErrorConstantRigCostFunctorWithNoise =
    IsotropicNoiseCostFunctorWrapper<
        RigReprojErrorConstantRigCostFunctor<CameraModel>>;

void BindCostFunctions(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("cost_functions");
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorCostFunctor, const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorCostFunctorWithNoise,
                            const double,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error with 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoseCostFunctorWithNoise,
                            const double,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose and 2D detection noise.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoint3DCostFunctor,
                            const Eigen::Vector2d&,
                            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point.");
  m.def("ReprojErrorCost",
        &CameraCostFunction<ReprojErrorConstantPoint3DCostFunctorWithNoise,
                            const double,
                            const Eigen::Vector2d&,
                            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point and 2D detection noise.");

  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorCostFunctor, const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error for camera rig.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorCostFunctorWithNoise,
                            const double,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error for camera rig with 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorConstantRigCostFunctor,
                            const Rigid3d&,
                            const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose.");
  m.def("RigReprojErrorCost",
        &CameraCostFunction<RigReprojErrorConstantRigCostFunctorWithNoise,
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
        &SampsonErrorCostFunctor::Create,
        "point2D1"_a,
        "point2D2"_a,
        "Sampson error for two-view geometry.");

  m.def("AbsolutePoseErrorCost",
        &AbsolutePoseErrorCostFunctor::Create,
        "cam_from_world"_a,
        "covariance_cam"_a,
        "6-DoF error on the absolute pose.");
  m.def("MetricRelativePoseErrorCost",
        &MetricRelativePoseErrorCostFunctor::Create,
        "i_from_j"_a,
        "covariance_j"_a,
        "6-DoF error between two absolute poses based on their relative pose.");
  m.def("Point3dAlignmentCost",
        &Point3dAlignmentCostFunctor::Create,
        "ref_point"_a,
        "covariance_point"_a,
        "Error between 3D points transformed by a similarity transform.");
  m.def("PositionPriorErrorCost",
        &PositionPriorErrorCostFunctor::Create,
        "world_from_cam_position_prior"_a,
        "covariance"_a);
}
