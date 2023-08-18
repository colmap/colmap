// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/types.h"

#include <memory>

namespace colmap {

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = default;

  // If the same matcher is used for matching multiple pairs of feature sets,
  // then the caller may pass a nullptr to one of the keypoint/descriptor
  // arguments to inform the implementation that the keypoints/descriptors are
  // identical to the previous call. This allows the implementation to skip e.g.
  // uploading data to GPU memory or pre-computing search data structures for
  // one of the descriptors.

  virtual void Match(
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      FeatureMatches* matches) = 0;

  virtual void MatchGuided(
      const TwoViewGeometryOptions& options,
      const std::shared_ptr<const FeatureKeypoints>& keypoints1,
      const std::shared_ptr<const FeatureKeypoints>& keypoints2,
      const std::shared_ptr<const FeatureDescriptors>& descriptors1,
      const std::shared_ptr<const FeatureDescriptors>& descriptors2,
      TwoViewGeometry* two_view_geometry) = 0;
};

}  // namespace colmap
