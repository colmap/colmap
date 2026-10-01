// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/cost_functions/alignment.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

// Create is overloaded on single and per-residual stddevs, so taking its
// address needs a target signature to disambiguate.
template <class CostFunctor, typename... Args>
ceres::CostFunction* (*ScaleCost())(double, Args&&...) {
  return &ScaleWeightedCostFunctor<CostFunctor>::template Create<Args...>;
}

template <class CostFunctor, typename... Args>
ceres::CostFunction* (*ScaleVecCost())(
    const typename ScaleWeightedCostFunctor<CostFunctor>::StddevVec&,
    Args&&...) {
  return &ScaleWeightedCostFunctor<CostFunctor>::template Create<Args...>;
}

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
        &CreateCovarianceWeightedCameraCostFunction<ReprojErrorCostFunctor,
                                                    Eigen::Matrix2d,
                                                    const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "Reprojection error with 2D detection noise.");
  m.def("ReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<ReprojErrorCostFunctor,
                                               double,
                                               const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "Reprojection error with isotropic 2D detection noise.");
  m.def("ReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<ReprojErrorCostFunctor,
                                               Eigen::Vector2d,
                                               const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "Reprojection error with per-axis 2D detection noise.");

  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                                  const Eigen::Vector2d&,
                                  const Rigid3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "cam_from_world"_a,
        "Reprojection error with constant camera pose.");
  m.def("ReprojErrorCost",
        &CreateCovarianceWeightedCameraCostFunction<
            ReprojErrorConstantPoseCostFunctor,
            Eigen::Matrix2d,
            const Eigen::Vector2d&,
            const Rigid3d&>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "cam_from_world"_a,
        "Reprojection error with constant camera pose and 2D detection noise.");
  m.def(
      "ReprojErrorCost",
      &CreateScaleWeightedCameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                                             double,
                                             const Eigen::Vector2d&,
                                             const Rigid3d&>,
      "camera_model_id"_a,
      "point2D_stddev"_a,
      "point2D"_a,
      "cam_from_world"_a,
      "Reprojection error with constant camera pose and isotropic 2D "
      "detection noise.");
  m.def(
      "ReprojErrorCost",
      &CreateScaleWeightedCameraCostFunction<ReprojErrorConstantPoseCostFunctor,
                                             Eigen::Vector2d,
                                             const Eigen::Vector2d&,
                                             const Rigid3d&>,
      "camera_model_id"_a,
      "point2D_stddev"_a,
      "point2D"_a,
      "cam_from_world"_a,
      "Reprojection error with constant camera pose and per-axis 2D "
      "detection noise.");

  m.def("ReprojErrorCost",
        &CreateCameraCostFunction<ReprojErrorConstantPoint3DCostFunctor,
                                  const Eigen::Vector2d&,
                                  const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point.");
  m.def("ReprojErrorCost",
        &CreateCovarianceWeightedCameraCostFunction<
            ReprojErrorConstantPoint3DCostFunctor,
            Eigen::Matrix2d,
            const Eigen::Vector2d&,
            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point and 2D detection noise.");
  m.def("ReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<
            ReprojErrorConstantPoint3DCostFunctor,
            double,
            const Eigen::Vector2d&,
            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point and isotropic 2D detection "
        "noise.");
  m.def("ReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<
            ReprojErrorConstantPoint3DCostFunctor,
            Eigen::Vector2d,
            const Eigen::Vector2d&,
            const Eigen::Vector3d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "point3D"_a,
        "Reprojection error with constant 3D point and per-axis 2D detection "
        "noise.");

  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorCostFunctor,
                                  const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "Reprojection error for camera rig.");
  m.def("RigReprojErrorCost",
        &CreateCovarianceWeightedCameraCostFunction<RigReprojErrorCostFunctor,
                                                    Eigen::Matrix2d,
                                                    const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "Reprojection error for camera rig with 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<RigReprojErrorCostFunctor,
                                               double,
                                               const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "Reprojection error for camera rig with isotropic 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<RigReprojErrorCostFunctor,
                                               Eigen::Vector2d,
                                               const Eigen::Vector2d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "Reprojection error for camera rig with per-axis 2D detection noise.");

  m.def("RigReprojErrorCost",
        &CreateCameraCostFunction<RigReprojErrorConstantRigCostFunctor,
                                  const Eigen::Vector2d&,
                                  const Rigid3d&>,
        "camera_model_id"_a,
        "point2D"_a,
        "cam_from_rig"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose.");
  m.def("RigReprojErrorCost",
        &CreateCovarianceWeightedCameraCostFunction<
            RigReprojErrorConstantRigCostFunctor,
            Eigen::Matrix2d,
            const Eigen::Vector2d&,
            const Rigid3d&>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "cam_from_rig"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose and "
        "2D detection noise.");
  m.def("RigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<
            RigReprojErrorConstantRigCostFunctor,
            double,
            const Eigen::Vector2d&,
            const Rigid3d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "cam_from_rig"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose and "
        "isotropic 2D detection noise.");
  m.def("RigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<
            RigReprojErrorConstantRigCostFunctor,
            Eigen::Vector2d,
            const Eigen::Vector2d&,
            const Rigid3d&>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "cam_from_rig"_a,
        "Reprojection error for camera rig with constant cam-from-rig pose and "
        "per-axis 2D detection noise.");

  m.def("ScaledRigReprojErrorCost",
        &CreateCameraCostFunction<ScaledRigReprojErrorCostFunctor,
                                  const Eigen::Vector2d&,
                                  bool>,
        "camera_model_id"_a,
        "point2D"_a,
        "use_log_scale"_a = true,
        "Reprojection error for camera rig with a scaled rig-from-world "
        "transform.");
  m.def("ScaledRigReprojErrorCost",
        &CreateCovarianceWeightedCameraCostFunction<
            ScaledRigReprojErrorCostFunctor,
            Eigen::Matrix2d,
            const Eigen::Vector2d&,
            bool>,
        "camera_model_id"_a,
        "point2D_cov"_a,
        "point2D"_a,
        "use_log_scale"_a = true,
        "Reprojection error for camera rig with a scaled rig-from-world "
        "transform and 2D detection noise.");
  m.def("ScaledRigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<ScaledRigReprojErrorCostFunctor,
                                               double,
                                               const Eigen::Vector2d&,
                                               bool>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "use_log_scale"_a = true,
        "Reprojection error for camera rig with a scaled rig-from-world "
        "transform and isotropic 2D detection noise.");
  m.def("ScaledRigReprojErrorCost",
        &CreateScaleWeightedCameraCostFunction<ScaledRigReprojErrorCostFunctor,
                                               Eigen::Vector2d,
                                               const Eigen::Vector2d&,
                                               bool>,
        "camera_model_id"_a,
        "point2D_stddev"_a,
        "point2D"_a,
        "use_log_scale"_a = true,
        "Reprojection error for camera rig with a scaled rig-from-world "
        "transform and per-axis 2D detection noise.");

  m.def("SampsonErrorCost",
        &SampsonErrorCostFunctor::Create<const Eigen::Vector2d&,
                                         const Eigen::Vector2d&>,
        "point1"_a,
        "point2"_a,
        "Sampson error for two-view geometry on image-plane points.");

  m.def("AbsolutePosePriorCost",
        &AbsolutePosePriorCostFunctor::Create<const Rigid3d&>,
        "cam_from_world_prior"_a,
        "6-DoF error on the absolute camera pose.");
  m.def("AbsolutePosePriorCost",
        &CovarianceWeightedCostFunctor<AbsolutePosePriorCostFunctor>::Create<
            const Rigid3d&>,
        "cam_cov_from_world_prior"_a,
        "cam_from_world_prior"_a,
        "6-DoF error on the absolute camera pose with prior covariance.");
  m.def("AbsolutePosePriorCost",
        ScaleVecCost<AbsolutePosePriorCostFunctor, const Rigid3d&>(),
        "cam_stddev_from_world_prior"_a,
        "cam_from_world_prior"_a,
        "6-DoF error on the absolute camera pose with per-DoF prior standard "
        "deviations. The first three are on the rotation and the last three on "
        "the translation, so they do not share a unit.");

  m.def("AbsolutePosePositionPriorCost",
        &AbsolutePosePositionPriorCostFunctor::Create<const Eigen::Vector3d&>,
        "position_in_world_prior"_a,
        "3-DoF error on the absolute camera pose's position.");
  m.def(
      "AbsolutePosePositionPriorCost",
      &CovarianceWeightedCostFunctor<
          AbsolutePosePositionPriorCostFunctor>::Create<const Eigen::Vector3d&>,
      "position_cov_in_world_prior"_a,
      "position_in_world_prior"_a,
      "3-DoF error on the absolute camera pose's position with prior "
      "covariance.");
  m.def(
      "AbsolutePosePositionPriorCost",
      ScaleCost<AbsolutePosePositionPriorCostFunctor, const Eigen::Vector3d&>(),
      "position_stddev_in_world_prior"_a,
      "position_in_world_prior"_a,
      "3-DoF error on the absolute camera pose's position with isotropic "
      "prior standard deviation.");
  m.def("AbsolutePosePositionPriorCost",
        ScaleVecCost<AbsolutePosePositionPriorCostFunctor,
                     const Eigen::Vector3d&>(),
        "position_stddev_in_world_prior"_a,
        "position_in_world_prior"_a,
        "3-DoF error on the absolute camera pose's position with per-axis "
        "prior standard deviations.");

  m.def("RelativePosePriorCost",
        &RelativePosePriorCostFunctor::Create<const Rigid3d&>,
        "i_from_j_prior"_a,
        "6-DoF error between two absolute camera poses based on a prior "
        "relative pose.");
  m.def("RelativePosePriorCost",
        &CovarianceWeightedCostFunctor<RelativePosePriorCostFunctor>::Create<
            const Rigid3d&>,
        "i_cov_from_j_prior"_a,
        "i_from_j_prior"_a,
        "6-DoF error between two absolute camera poses based on a prior "
        "relative pose with prior covariance.");
  m.def("RelativePosePriorCost",
        ScaleVecCost<RelativePosePriorCostFunctor, const Rigid3d&>(),
        "i_stddev_from_j_prior"_a,
        "i_from_j_prior"_a,
        "6-DoF error between two absolute camera poses based on a prior "
        "relative pose with per-DoF prior standard deviations. The first three "
        "are on the rotation and the last three on the translation, so they do "
        "not share a unit.");

  m.def("Point3DAlignmentCost",
        &Point3DAlignmentCostFunctor::Create<const Eigen::Vector3d&, bool>,
        "point_in_b_prior"_a,
        "use_log_scale"_a = true,
        "Error between 3D points transformed by a 3D similarity transform.");
  m.def("Point3DAlignmentCost",
        &CovarianceWeightedCostFunctor<
            Point3DAlignmentCostFunctor>::Create<const Eigen::Vector3d&, bool>,
        "point_cov_in_b_prior"_a,
        "point_in_b_prior"_a,
        "use_log_scale"_a = true,
        "Error between 3D points transformed by a 3D similarity transform. "
        "with prior covariance");
  m.def("Point3DAlignmentCost",
        ScaleCost<Point3DAlignmentCostFunctor, const Eigen::Vector3d&, bool>(),
        "point_stddev_in_b_prior"_a,
        "point_in_b_prior"_a,
        "use_log_scale"_a = true,
        "Error between 3D points transformed by a 3D similarity transform, "
        "with an isotropic prior standard deviation.");
  m.def(
      "Point3DAlignmentCost",
      ScaleVecCost<Point3DAlignmentCostFunctor, const Eigen::Vector3d&, bool>(),
      "point_stddev_in_b_prior"_a,
      "point_in_b_prior"_a,
      "use_log_scale"_a = true,
      "Error between 3D points transformed by a 3D similarity transform, "
      "with per-axis prior standard deviations.");
}
