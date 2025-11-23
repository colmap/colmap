#pragma once

#include "glomap/scene/camera.h"
#include "glomap/scene/types.h"
#include "glomap/types.h"

namespace glomap {

// Cheirality check for essential matrix
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_depth = 0.,
                     double max_depth = 100.);

// Get the orientation signum for fundamental matrix
// For chierality check of fundamental matrix
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2);

// Get the essential matrix from relative pose
void EssentialFromMotion(const Rigid3d& pose, Eigen::Matrix3d* E);

// Get the essential matrix from relative pose
void FundamentalFromMotionAndCameras(const Camera& camera1,
                                     const Camera& camera2,
                                     const Rigid3d& pose,
                                     Eigen::Matrix3d* F);

// Sampson error for the essential matrix
// Input the normalized image coordinates (2d)
double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector2d& x1,
                    const Eigen::Vector2d& x2);

// Sampson error for the essential matrix
// Input the normalized image ray (3d)
double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector3d& x1,
                    const Eigen::Vector3d& x2);

// Homography error for the homography matrix
double HomographyError(const Eigen::Matrix3d& H,
                       const Eigen::Vector2d& x1,
                       const Eigen::Vector2d& x2);

}  // namespace glomap
