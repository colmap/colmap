// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_RETRIEVAL_VOTE_AND_VERIFY_H_
#define COLMAP_SRC_RETRIEVAL_VOTE_AND_VERIFY_H_

#include "retrieval/geometry.h"

namespace colmap {
namespace retrieval {

struct VoteAndVerifyOptions {
  // Number of top transformations to generate.
  int num_transformations = 30;

  // Number of voting bins in the translation dimension.
  int num_trans_bins = 64;

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
