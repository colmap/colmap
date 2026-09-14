// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <optional>
#include <vector>

namespace colmap {
namespace mvs {

// Interface for estimators that compute the depth and normal map of a
// reference image from a set of source images.
class MVSEstimator {
 public:
  MAKE_ENUM_CLASS(Type, 0, PATCH_MATCH);

  // The photometric pass estimates the maps from the images alone. The
  // geometric pass additionally enforces consistency with the maps of a
  // previous photometric pass over all images.
  enum class Pass { PHOTOMETRIC, GEOMETRIC };

  struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth and normal maps for the geometric pass.
    std::vector<DepthMap>* depth_maps = nullptr;
    std::vector<NormalMap>* normal_maps = nullptr;

    // Depth range of the reference image, which must be positive. Estimators
    // ignore the depth range in their options, which callers use to set up
    // the problem.
    float depth_min = -1.0f;
    float depth_max = -1.0f;

    // Print the configuration to stdout.
    void Print() const;
  };

  struct Result {
    DepthMap depth_map;
    NormalMap normal_map;

    // Only set if enabled in the estimator options.
    std::optional<ConsistencyGraph> consistency_graph;
  };

  struct Capabilities {
    // Whether the estimator expects RGB images instead of grayscale images.
    bool requires_rgb = false;

    // Whether the estimator supports the geometric pass.
    bool supports_geometric_pass = true;

    // Whether a problem may list the same source image multiple times, e.g.,
    // to pad it to the minimum number of source images.
    bool supports_repeated_source_images = false;

    // Supported number of source images per problem.
    size_t min_num_source_images = 1;
    std::optional<size_t> max_num_source_images;
  };

  struct Options;

  virtual ~MVSEstimator() = default;

  // Create the estimator selected by the options on the given device, which
  // overrides the GPU index in the options.
  static std::unique_ptr<MVSEstimator> Create(const Options& options,
                                              int device_index);

  virtual Capabilities GetCapabilities() const = 0;
  virtual Result Estimate(const Problem& problem, Pass pass) = 0;
};

struct MVSEstimator::Options {
  explicit Options(Type type = DefaultType());

  static Type DefaultType();

  std::shared_ptr<PatchMatchOptions> patch_match;
  Type type;

  Options(const Options& other);
  Options& operator=(const Options& other);
  Options(Options&& other) = default;
  Options& operator=(Options&& other) = default;

  bool Check() const;
};

}  // namespace mvs
}  // namespace colmap
