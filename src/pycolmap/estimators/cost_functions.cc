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

  m.def(
      "ReprojErrorCost",
      &CreateCameraCostFunction<ReprojErrorCostFunctor, const Eigen::Vector2d&>,
      "camera_model_id"_a,
      "point2D"_a,
      "Reprojection error.");
  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorCostFunctorWithNoise,
                                  const double,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error with 2D detection noise.");
  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                                  const Rigid3d&,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose.");
  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctorWithNoise,
                                  const double,
                                  const Rigid3d&,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "cam_from_world"_a,
        "point2D"_a,
        "Reprojection error with constant camera pose and 2D detection noise.");
  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorConstantPoint3DCostFunctor,
                                  const Eigen::Vector2d&,
                                  const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point.");
  m.def(
      "ReprojErrorCost",
      &CreateCameraCostFunction<ReprojErrorConstantPoint3DCostFunctorWithNoise,
                                const double,
                                const Eigen::Vector2d&,
                                const Eigen::Vector3d&>,
      "camera_model_id"_a,
      "stddev"_a,
      "point2D"_a,
      "point3D"_a,
      "Reprojection error with constant 3D point and 2D detection noise.");

  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorCostFunctor,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error for camera rig.");
  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorCostFunctorWithNoise,
                                  const double,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "stddev"_a,
        "point2D"_a,
        "Reprojection error for camera rig with 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor,
                                  const Rigid3d&,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "cam_from_rig"_a,
        "point2D"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose.");
  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctorWithNoise,
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
        &SampsonErrorCostFunctor::Create<const Eigen::Vector2d&,
                                         const Eigen::Vector2d&>,
        "point2D1"_a,
        "point2D2"_a,
        "Sampson error for two-view geometry.");

  m.def("AbsolutePosePriorCost",
        &AbsolutePosePriorCostFunctor::Create<const Rigid3d&,
                                              const Eigen::Matrix6d&>,
        "cam_from_world_prior"_a,
        "cam_cov_from_world_prior"_a,
        "6-DoF error on the absolute camera pose.");
  m.def("AbsolutePosePositionPriorCost",
        &AbsolutePosePositionPriorCostFunctor::Create<const Eigen::Vector3d&&,
                                                      const Eigen::Matrix3d&>,
        "position_in_world_prior"_a,
        "position_cov_in_world_prior"_a,
        "3-DoF error on the absolute camera pose's position.");
  m.def("RelativePosePriorCost",
        &RelativePosePriorCostFunctor::Create<const Rigid3d&,
                                              const Eigen::Matrix6d&>,
        "i_from_j_prior"_a,
        "i_cov_from_j_prior"_a,
        "6-DoF error between two absolute camera poses based on a prior "
        "relative pose.");
  m.def("Point3DAlignmentCost",
        &Point3DAlignmentCostFunctor::Create<const Eigen::Vector3d&&,
                                             const Eigen::Matrix3d&>,
        "point_in_b_prior"_a,
        "point_cov_in_b_prior"_a,
        "Error between 3D points transformed by a 3D similarity transform.");
}
