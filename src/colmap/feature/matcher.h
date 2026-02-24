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

#include "colmap/feature/types.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/types.h"

#include <memory>
#include <string>

namespace colmap {

struct SiftMatchingOptions;
struct AlikedMatchingOptions;

struct FeatureMatchingTypeOptions {
  explicit FeatureMatchingTypeOptions();

  std::shared_ptr<SiftMatchingOptions> sift;
  std::shared_ptr<AlikedMatchingOptions> aliked;

  FeatureMatchingTypeOptions(const FeatureMatchingTypeOptions& other);
  FeatureMatchingTypeOptions& operator=(
      const FeatureMatchingTypeOptions& other);
  FeatureMatchingTypeOptions(FeatureMatchingTypeOptions&& other) = default;
  FeatureMatchingTypeOptions& operator=(FeatureMatchingTypeOptions&& other) =
      default;
};

struct FeatureMatchingOptions : public FeatureMatchingTypeOptions {
  explicit FeatureMatchingOptions(
      FeatureMatcherType type = FeatureMatcherType::SIFT_BRUTEFORCE);

  FeatureMatcherType type = FeatureMatcherType::SIFT_BRUTEFORCE;

  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
#ifdef COLMAP_GPU_ENABLED
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Whether to perform guided matching.
  bool guided_matching = false;

  // Skips the geometric verification stage and forwards matches unchanged.
  // This option is ignored when guided matching is enabled, because guided
  // matching depends on the two-view geometry produced by geometric
  // verification.
  bool skip_geometric_verification = false;

  // Whether to perform geometric verification using rig constraints
  // between pairs of non-trivial frames. If disabled, performs geometric
  // two-view verification for non-trivial frames without rig constraints.
  // This option is ignored when skip_geometric_verification is true.
  bool rig_verification = false;

  // Whether to skip matching images within the same frame.
  // This is useful for the case of non-overlapping cameras in a rig.
  bool skip_image_pairs_in_same_frame = false;

  // Whether the selected matcher requires OpenGL.
  bool RequiresOpenGL() const;

  bool Check() const;
};

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = default;

  struct Image {
    // Unique identifier for the image. Allows a matcher to cache some
    // computations per image in consecutive calls to matching.
    image_t image_id = kInvalidImageId;
    const Camera* camera = nullptr;
    std::shared_ptr<const FeatureKeypoints> keypoints;
    std::shared_ptr<const FeatureDescriptors> descriptors;
    const PosePrior* pose_prior = nullptr;
  };

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options);

  virtual void Match(const Image& image1,
                     const Image& image2,
                     FeatureMatches* matches) = 0;

  virtual void MatchGuided(double max_error,
                           const Image& image1,
                           const Image& image2,
                           TwoViewGeometry* two_view_geometry) = 0;
};

}  // namespace colmap
