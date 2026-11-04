// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/cusift.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <cmath>

#ifdef COLMAP_CUDA_ENABLED
// Deliberately no <cuda_runtime.h> and no cudautils.h here, same rule as
// cusift.cc: every CUDA type and API reference stays inside colmap_cusift
// (see cusift_device.h), so colmap_feature acquires no CUDA runtime
// dependency of its own.
#include "cusift_device.h"
// Angle brackets deliberately: this directory (src/colmap/feature/) also has
// its own matcher.h (colmap's FeatureMatcher interface), and a quoted include
// would resolve to that one first (quoted-include lookup checks the
// including file's own directory before the include path) rather than the
// vendored src/thirdparty/cusift/src/matcher.h.
#include <matcher.h>  // vendored: ::Match{i,j,dist2}, ::MatchDescriptorsGemmU8
#endif                // COLMAP_CUDA_ENABLED

namespace colmap {
namespace {

#ifdef COLMAP_CUDA_ENABLED

// No host-side descriptor conversion at all: COLMAP stores SIFT descriptors as
// row-major uint8, which is exactly what MatchDescriptorsGemmU8 uploads. That
// keeps the transfer at 128 B/descriptor and moves the widening onto the
// device.
//
// Deliberately NOT L2-normalized. COLMAP's reference matcher ranks candidates
// on the RAW inner product and converts with a FIXED 512^2 constant rather than
// per-row norms (sift.cc, FindBestMatchesOneWayBruteForce); SiftGPU does the
// same. Normalizing per row would re-rank near-ties and lose bit-exactness
// with that reference.

class CusiftFeatureMatcher : public FeatureMatcher {
 public:
  explicit CusiftFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options) {
    THROW_CHECK(THROW_CHECK_NOTNULL(options_.sift)->Check() &&
                THROW_CHECK_NOTNULL(options_.cusift)->Check())
        << "invalid SIFT/cusift matching options; the reason is logged above";
    THROW_CHECK(!options_.guided_matching)
        << "cusift GEMM matcher does not implement guided matching; use "
           "SIFT_BRUTEFORCE or SIFT_LIGHTGLUE for guided matching, or disable "
           "guided_matching.";
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    BindCusiftDevice(options.gpu_index);
    // Size the device buffers once, on this worker's thread, before any pair is
    // matched. FeatureMatcherWorker has already clamped max_num_matches to the
    // cache's largest keypoint count (feature_matching_utils.cc), so it is
    // exactly the per-side upper bound. Without this the buffers grow on demand
    // and each growth calls cudaFree, which synchronises the whole device and
    // stalls whatever else COLMAP has queued.
    ReserveGemmMatcher(options.max_num_matches);
    return std::make_unique<CusiftFeatureMatcher>(options);
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    THROW_CHECK_NOTNULL(image1.descriptors);
    THROW_CHECK_NOTNULL(image2.descriptors);
    THROW_CHECK_EQ(image1.descriptors->type, FeatureExtractorType::SIFT);
    THROW_CHECK_EQ(image2.descriptors->type, FeatureExtractorType::SIFT);

    matches->clear();
    const int n1 = static_cast<int>(image1.descriptors->data.rows());
    const int n2 = static_cast<int>(image2.descriptors->data.rows());
    if (n1 == 0 || n2 == 0) {
      return;
    }

    const bool angular =
        options_.cusift->distance_type == CusiftDistanceType::ANGULAR;
    // COLMAP's fixed descriptor-norm convention (sift.cc
    // kSqSiftDescriptorNorm).
    constexpr double kSqSiftDescriptorNorm = 512.0 * 512.0;

    std::vector<::Match> raw_matches;
    MatchDescriptorsGemmU8(
        image1.descriptors->data.data(),
        n1,
        image2.descriptors->data.data(),
        n2,
        static_cast<float>(options_.sift->max_ratio),
        options_.sift->cross_check,
        raw_matches,
        options_.cusift->cross_check_reverse_ratio,
        angular ? kGemmDistanceAngular : kGemmDistanceL2,
        // ANGULAR wants radians; L2 wants descriptor-scale distance, so lift
        // the unit-vector threshold onto COLMAP's 512 scale.
        static_cast<float>(angular ? options_.sift->max_distance
                                   : 512.0 * options_.sift->max_distance),
        static_cast<float>(kSqSiftDescriptorNorm));

    // max_num_matches post-filter: replicate SiftMatchGPU's cap-binds rule
    // (SiftMatchCU::GetBestMatch) -- ascending image1-index order, first-N
    // survive, NOT a score-sorted top-K. MatchDescriptorsGemm's row loop
    // already produces raw_matches in ascending `i` order (matcher_gemm.cu),
    // so this is a straight truncation, not a re-sort.
    if (raw_matches.size() > static_cast<size_t>(options_.max_num_matches)) {
      raw_matches.resize(static_cast<size_t>(options_.max_num_matches));
    }

    matches->reserve(raw_matches.size());
    for (const ::Match& m : raw_matches) {
      matches->emplace_back(static_cast<point2D_t>(m.i),
                            static_cast<point2D_t>(m.j));
    }
  }

  void MatchGuided(double /*max_error*/,
                   const Image& /*image1*/,
                   const Image& /*image2*/,
                   TwoViewGeometry* /*two_view_geometry*/) override {
    THROW_CHECK(false)
        << "cusift GEMM matcher does not implement guided matching.";
  }

 private:
  const FeatureMatchingOptions options_;
};

#endif  // COLMAP_CUDA_ENABLED

}  // namespace

bool CusiftMatchingOptions::Check() const { return true; }

std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_CUDA_ENABLED
  LOG(INFO) << "Creating cusift GEMM feature matcher";
  return CusiftFeatureMatcher::Create(options);
#else
  LOG(ERROR) << "cusift feature matching requires CUDA support";
  return nullptr;
#endif  // COLMAP_CUDA_ENABLED
}

}  // namespace colmap
