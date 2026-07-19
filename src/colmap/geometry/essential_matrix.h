// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Decompose an essential matrix into the possible rotations and translations.
//
// The first pose is assumed to be P = [I | 0] and the set of four other
// possible second poses are defined as: {[R1 | t], [R2 | t],
//                                        [R1 | -t], [R2 | -t]}
//
// @param E          3x3 essential matrix.
// @param R1         First possible 3x3 rotation matrix.
// @param R2         Second possible 3x3 rotation matrix.
// @param t          3x1 possible translation vector (also -t possible).
void DecomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2,
                              Eigen::Vector3d* t);

// Recover the most probable pose from the given essential matrix.
//
// The pose of the first image is assumed to be P = [I | 0].
//
// @param E               3x3 essential matrix.
// @param cam_rays1       First set of corresponding rays.
// @param cam_rays2       Second set of corresponding rays.
// @param cam2_from_cam1  Relative camera transformation.
// @param valid_indices   Indices of correspondences in front of both cameras.
void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector3d>& cam_rays1,
                             const std::vector<Eigen::Vector3d>& cam_rays2,
                             Rigid3d* cam2_from_cam1,
                             std::vector<int>* valid_indices);

// Compose essential matrix from relative camera poses.
//
// Assumes that first camera pose has projection matrix P = [I | 0], and
// pose of second camera is given as transformation from world to camera system.
//
// @param cam2_from_cam1  Relative camera transformation.
//
// @return                3x3 essential matrix.
Eigen::Matrix3d EssentialMatrixFromPose(const Rigid3d& cam2_from_cam1);

// Find optimal image points, such that:
//
//     optimal_point1^t * E * optimal_point2 = 0
//
// as described in:
//
//   Lindstrom, P., "Triangulation made easy",
//   Computer Vision and Pattern Recognition (CVPR),
//   2010 IEEE Conference on , vol., no., pp.1554,1561, 13-18 June 2010
//
// @param E                Essential or fundamental matrix.
// @param point1           Corresponding 2D point in first image.
// @param point2           Corresponding 2D point in second image.
// @param optimal_point1   Estimated optimal image point in the first image.
// @param optimal_point2   Estimated optimal image point in the second image.
void FindOptimalImageObservations(const Eigen::Matrix3d& E,
                                  const Eigen::Vector2d& point1,
                                  const Eigen::Vector2d& point2,
                                  Eigen::Vector2d* optimal_point1,
                                  Eigen::Vector2d* optimal_point2);

// Compute the location of the epipole in homogeneous coordinates.
//
// @param E           3x3 essential matrix.
// @param left_image  If true, epipole in left image is computed,
//                    else in right image.
//
// @return            Epipole in homogeneous coordinates.
Eigen::Vector3d EpipoleFromEssentialMatrix(const Eigen::Matrix3d& E,
                                           bool left_image);

// Invert the essential matrix, i.e. if the essential matrix E describes the
// transformation from camera A to B, the inverted essential matrix E' describes
// the transformation from camera B to A.
//
// @param E      3x3 essential matrix.
//
// @return       Inverted essential matrix.
Eigen::Matrix3d InvertEssentialMatrix(const Eigen::Matrix3d& matrix);

// Composes the fundamental matrix from image 1 to 2 from the essential matrix
// and two camera's calibrations.
Eigen::Matrix3d FundamentalFromEssentialMatrix(const Eigen::Matrix3d& K2,
                                               const Eigen::Matrix3d& E,
                                               const Eigen::Matrix3d& K1);

// Composes the essential matrix from image 1 to 2 from the fundamental matrix
// and two camera's calibrations.
Eigen::Matrix3d EssentialFromFundamentalMatrix(const Eigen::Matrix3d& K2,
                                               const Eigen::Matrix3d& F,
                                               const Eigen::Matrix3d& K1);

// Calculate the squared Sampson error for a single point pair and a given
// fundamental or essential matrix.
//
// @param ray1        First point/ray in homogeneous coordinates.
// @param ray2        Second point/ray in homogeneous coordinates.
// @param E           3x3 fundamental or essential matrix.
// @return            Squared Sampson error.
double ComputeSquaredSampsonError(const Eigen::Vector3d& ray1,
                                  const Eigen::Vector3d& ray2,
                                  const Eigen::Matrix3d& E);

// Calculate the residuals of a set of corresponding points and a given
// fundamental or essential matrix.
//
// Residuals are defined as the squared Sampson error.
//
// @param points1     Corresponding points.
// @param points2     Corresponding points.
// @param E           3x3 fundamental or essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals);

// Calculate the residuals of a set of corresponding rays and a given
// fundamental or essential matrix.
//
// Residuals are defined as the squared Sampson error.
//
// @param rays1       Corresponding rays.
// @param rays2       Corresponding rays.
// @param E           3x3 fundamental or essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredSampsonError(const std::vector<Eigen::Vector3d>& rays1,
                                const std::vector<Eigen::Vector3d>& rays2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals);

// Calculate the residuals of a set of corresponding rays and a given essential
// matrix, additionally enforcing the cheirality constraint.
//
// Residuals are the squared Sampson error, except that correspondences which
// triangulate behind either camera are assigned an infinite residual. The
// relative pose is recovered from E (resolving the four-fold decomposition
// ambiguity by cheirality voting), so an epipolar-consistent correspondence
// with the wrong depth sign is rejected even when its Sampson error is small.
//
// Only meaningful for essential matrices (calibrated rays); the plain
// ComputeSquaredSampsonError should be used for fundamental matrices.
//
// @param rays1       Corresponding rays.
// @param rays2       Corresponding rays.
// @param E           3x3 essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredSampsonErrorWithCheirality(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals);

// Calculate the squared tangent Sampson error for a single ray pair and a given
// essential matrix.
//
// The Sampson approximation is C(z)^2 / ||dC/dz||^2 for a constraint C and
// measurements z. Taking z to be the *pixel* coordinates, rather than the rays,
// yields an error in pixel units for any central camera model:
//
//     C            = ray2^T E ray1
//     dC/dpx1      = J1^T (E^T ray2)
//     dC/dpx2      = J2^T (E ray1)
//     error        = C^2 / (||dC/dpx1||^2 + ||dC/dpx2||^2)
//
// where J = d(ray) / d(pixel) is the unprojection Jacobian, obtainable via
// Camera::CamRayFromImgWithJac. This is the tangent Sampson error of Terekhov
// and Larsson, "Tangent Sampson Error: Fast Approximate Two-view Reprojection
// Error for Central Camera Models", ICCV 2023.
//
// Pixels are the space in which feature detection noise is (approximately)
// isotropic and uniform, so a threshold on this residual is meaningful in
// pixels across the whole image and for every camera model - unlike the plain
// Sampson error on unit bearings, whose pixel-equivalent tolerance grows with
// the angle from the principal direction.
//
// Note that the Sampson approximation is not invariant to the choice of
// homogeneous representative when that choice varies with the measurements:
// rescaling the constraint by g(z) perturbs the result by a term proportional
// to the residual. For an undistorted pinhole this reduces *exactly* to f^2
// times the Sampson error on normalized image coordinates when the (u, v, 1)
// representative is used (its Jacobian being the constant 1/f), and to first
// order in the residual when unit bearings are used. The latter is the sense in
// which this is an approximation of the true reprojection error.
//
// @param ray1        First unit bearing vector.
// @param J_ray1      Jacobian d(ray1) / d(pixel1).
// @param ray2        Second unit bearing vector.
// @param J_ray2      Jacobian d(ray2) / d(pixel2).
// @param E           3x3 essential matrix.
// @return            Squared tangent Sampson error, in squared pixels.
double ComputeSquaredTangentSampsonError(
    const Eigen::Vector3d& ray1,
    const Eigen::Matrix<double, 3, 2>& J_ray1,
    const Eigen::Vector3d& ray2,
    const Eigen::Matrix<double, 3, 2>& J_ray2,
    const Eigen::Matrix3d& E);

// Calculate the residuals of a set of corresponding rays and a given essential
// matrix, as the squared tangent Sampson error.
//
// @param rays1       Corresponding unit bearing vectors.
// @param J_rays1     Corresponding Jacobians d(ray1) / d(pixel1).
// @param rays2       Corresponding unit bearing vectors.
// @param J_rays2     Corresponding Jacobians d(ray2) / d(pixel2).
// @param E           3x3 essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredTangentSampsonError(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Matrix<double, 3, 2>>& J_rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const std::vector<Eigen::Matrix<double, 3, 2>>& J_rays2,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals);

// Calculate the residuals of a set of corresponding rays and a given essential
// matrix as the squared tangent Sampson error, additionally enforcing the
// cheirality constraint.
//
// Correspondences that triangulate behind either camera are assigned an
// infinite residual, as in ComputeSquaredSampsonErrorWithCheirality.
//
// @param rays1       Corresponding unit bearing vectors.
// @param J_rays1     Corresponding Jacobians d(ray1) / d(pixel1).
// @param rays2       Corresponding unit bearing vectors.
// @param J_rays2     Corresponding Jacobians d(ray2) / d(pixel2).
// @param E           3x3 essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredTangentSampsonErrorWithCheirality(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Matrix<double, 3, 2>>& J_rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const std::vector<Eigen::Matrix<double, 3, 2>>& J_rays2,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals);

}  // namespace colmap
