// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Decompose an homography matrix into the possible rotations, translations, and
// plane normal vectors, according to:
//
//    Malis, Ezio, and Manuel Vargas. "Deeper understanding of the homography
//    decomposition for vision-based control." (2007): 90.
//
// The first pose is assumed to be P = [I | 0]. Note that the homography is
// plane-induced if `cams2_from_cams1.size() == n.size() == 4`. If
// `cams2_from_cams1.size() == n.size() == 1` the homography is pure-rotational.
//
// @param H                 3x3 homography matrix.
// @param K1                3x3 calibration matrix of first camera.
// @param K2                3x3 calibration matrix of second camera.
// @param cams2_from_cams1  Possible relative camera transformations.
// @param normals           Possible normal vectors.
void DecomposeHomographyMatrix(const Eigen::Matrix3d& H,
                               const Eigen::Matrix3d& K1,
                               const Eigen::Matrix3d& K2,
                               std::vector<Rigid3d>* cams2_from_cams1,
                               std::vector<Eigen::Vector3d>* normals);

// Recover the most probable pose from the given homography matrix.
//
// The pose of the first image is assumed to be P = [I | 0].
//
// @param H               3x3 homography matrix.
// @param K1              3x3 calibration matrix of first camera.
// @param K2              3x3 calibration matrix of second camera.
// @param cam_rays1       First set of corresponding rays.
// @param cam_rays2       Second set of corresponding rays.
// @param cam2_from_cam1  Most probable relative camera transformation.
// @param normal          Most probable 3x1 normal vector.
// @param points3D        Triangulated 3D points infront of camera
//                        (only if homography is not pure-rotational).
void PoseFromHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2,
                              const std::vector<Eigen::Vector3d>& cam_rays1,
                              const std::vector<Eigen::Vector3d>& cam_rays2,
                              Rigid3d* cam2_from_cam1,
                              Eigen::Vector3d* normal,
                              std::vector<Eigen::Vector3d>* points3D);

// Compose homography matrix from relative pose.
//
// @param K1      3x3 calibration matrix of first camera.
// @param K2      3x3 calibration matrix of second camera.
// @param R       Most probable 3x3 rotation matrix.
// @param t       Most probable 3x1 translation vector.
// @param n       Most probable 3x1 normal vector.
// @param d       Orthogonal distance from plane.
//
// @return        3x3 homography matrix.
Eigen::Matrix3d HomographyMatrixFromPose(const Eigen::Matrix3d& K1,
                                         const Eigen::Matrix3d& K2,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& n,
                                         double d);

// Calculate the squared reprojection error for a single point pair under
// a homography transformation.
//
// @param point1      First point.
// @param point2      Second point.
// @param H           3x3 homography matrix.
// @return            Squared reprojection error.
double ComputeSquaredHomographyError(const Eigen::Vector2d& point1,
                                     const Eigen::Vector2d& point2,
                                     const Eigen::Matrix3d& H);

}  // namespace colmap
