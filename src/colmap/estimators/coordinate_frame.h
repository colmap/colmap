// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"

#include <filesystem>

namespace colmap {

struct ManhattanWorldFrameEstimationOptions {
  // The maximum image size for line detection.
  int max_image_size = 1024;
  // The minimum length of line segments in pixels.
  double min_line_length = 3;
  // The tolerance for classifying lines into horizontal/vertical.
  double line_orientation_tolerance = 0.2;
  // The maximum distance in pixels between lines and the vanishing points.
  double max_line_vp_distance = 0.5;
  // The maximum cosine distance between estimated axes to be inliers.
  double max_axis_distance = 0.05;
};

// Estimate gravity vector by assuming gravity-aligned image orientation, i.e.
// the majority of images is assumed to have the gravity vector aligned with an
// upright image plane.
Eigen::Vector3d EstimateGravityVectorFromImageOrientation(
    const Reconstruction& reconstruction, double max_axis_distance = 0.05);

// Estimate the coordinate frame of the reconstruction assuming a Manhattan
// world by finding the major vanishing points in each image. This function
// assumes that the majority of images is taken in upright direction, i.e.
// people are standing upright in the image. The orthonormal axes of the
// estimated coordinate frame will be given in the columns of the returned
// matrix. If one axis could not be determined, the respective column will be
// zero. The axes are specified in the world coordinate system in the order
// rightward, downward, forward.
#ifdef COLMAP_LSD_ENABLED
Eigen::Matrix3d EstimateManhattanWorldFrame(
    const ManhattanWorldFrameEstimationOptions& options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path);
#endif

// Aligns the reconstruction to the plane defined by running PCA on the 3D
// points. The model centroid is at the origin of the new coordinate system
// and the X axis is the first principal component with the Y axis being the
// second principal component
void AlignToPrincipalPlane(Reconstruction* recon, Sim3d* tform);

// Aligns the reconstruction to the local ENU plane orientation. Rotates the
// reconstruction such that the x-y plane aligns with the ENU tangent plane at
// the point cloud centroid and translates the origin to the centroid.
// If unscaled == true, then the original scale of the model remains unchanged.
void AlignToENUPlane(Reconstruction* recon, Sim3d* tform, bool unscaled);

}  // namespace colmap
