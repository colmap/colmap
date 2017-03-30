// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_
#define COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_

#include <Eigen/Core>

#include "base/reconstruction.h"

namespace colmap {

struct CoordinateFrameEstimationOptions {
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

// Estimate the coordinate frame of the reconstruction assuming a Manhattan
// world by finding the major vanishing points in each image. This function
// assumes that the majority of images is taken in upright direction, i.e.
// people are standing upright in the image. The orthonormal axes of the
// estimated coordinate frame will be given in the columns of the returned
// matrix. If one axis could not be determined, the respective column will be
// zero. The axes are specified in the world coordinate system in the order
// rightward, downward, forward.
Eigen::Matrix3d EstimateCoordinateFrame(
    const CoordinateFrameEstimationOptions& options,
    const Reconstruction& reconstruction, const std::string& image_path);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_COORDINATE_AXES_H_
