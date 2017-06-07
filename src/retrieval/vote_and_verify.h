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

#ifndef COLMAP_SRC_RETRIEVAL_VOTE_AND_VERIFY_H_
#define COLMAP_SRC_RETRIEVAL_VOTE_AND_VERIFY_H_

#include "retrieval/geometry.h"

namespace colmap {
namespace retrieval {

struct VoteAndVerifyOptions {
  // Number of top transformations to generate.
  int num_transformations = 10;

  // Number of voting bins in the translation dimension.
  int num_trans_bins = 32;

  // Number of voting bins in the scale dimension.
  int num_scale_bins = 32;

  // Number of voting bins in the orientation dimension.
  int num_angle_bins = 8;

  // Maximum image dimension that bounds the range of the translation bins.
  int max_image_size = 4096;

  // Minimum number of votes for a transformation to be considered.
  int min_num_votes = 1;

  // RANSAC confidence level used to abort the iteration.
  double confidence = 0.99;

  // Thresholds for considering a match an inlier.
  double max_transfer_error = 100.0 * 100.0;
  double max_scale_error = 2.0;
};

// Compute effective inlier count using Vote-and-Verify by estimating an affine
// transformation from 2D-2D image matches. The method is described in:
//      "A Vote­-and­-Verify Strategy for
//       Fast Spatial Verification in Image Retrieval",
//      Schönberger et al., ACCV 2016.
int VoteAndVerify(const VoteAndVerifyOptions& options,
                  const std::vector<FeatureGeometryMatch>& matches);

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_VOTE_AND_VERIFY_H_
