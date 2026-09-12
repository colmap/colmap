// SPDX-License-Identifier: BSD-3-Clause

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
struct LomaMatchingOptions;

struct FeatureMatchingTypeOptions {
  explicit FeatureMatchingTypeOptions();

  std::shared_ptr<SiftMatchingOptions> sift;
  std::shared_ptr<AlikedMatchingOptions> aliked;
  std::shared_ptr<LomaMatchingOptions> loma;

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
