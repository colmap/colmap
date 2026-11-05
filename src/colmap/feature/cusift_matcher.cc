// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/cusift.h"
#include "colmap/feature/sift.h"
#include "colmap/util/logging.h"

#include <vector>

#ifdef COLMAP_CUSIFT_ENABLED
// No CUDA headers: colmap_feature takes no CUDA runtime dependency.
#include "cusift_device.h"
#include "matcher_gemm.h"
#endif  // COLMAP_CUSIFT_ENABLED

namespace colmap {
namespace {

#ifdef COLMAP_CUSIFT_ENABLED

// Uploads COLMAP's row-major uint8 descriptors as they are, without per-row
// normalization: the reference (FindBestMatchesOneWayBruteForce) ranks on the
// raw inner product with a fixed 512^2 norm, and normalizing would re-rank
// near-ties.

class CusiftFeatureMatcher : public FeatureMatcher {
 public:
  explicit CusiftFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options) {
    // Includes use_gpu and guided_matching, for callers that skip Check().
    THROW_CHECK(options_.Check())
        << "invalid options for the cusift matcher; the reason is logged above";
  }

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    BindCusiftDevice(options.gpu_index);
    auto matcher = std::make_unique<CusiftFeatureMatcher>(options);
    // Size the device buffers once: the worker has clamped max_num_matches to
    // the largest keypoint count (feature_matching_utils.cc), and growing them
    // later calls cudaFree, which synchronises the whole device.
    matcher->gemm_.Reserve(options.max_num_matches);
    return matcher;
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    THROW_CHECK_NOTNULL(image1.descriptors);
    THROW_CHECK_NOTNULL(image2.descriptors);
    THROW_CHECK_EQ(image1.descriptors->type, FeatureExtractorType::SIFT);
    THROW_CHECK_EQ(image2.descriptors->type, FeatureExtractorType::SIFT);
    // The upload reads 128 bytes per row.
    THROW_CHECK_EQ(image1.descriptors->data.cols(), 128);
    THROW_CHECK_EQ(image2.descriptors->data.cols(), 128);

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
    gemm_.MatchPair(
        image1.descriptors->data.data(),
        n1,
        image2.descriptors->data.data(),
        n2,
        static_cast<float>(options_.sift->max_ratio),
        options_.sift->cross_check,
        raw_matches,
        options_.cusift->cross_check_reverse_ratio,
        angular ? kGemmDistanceAngular : kGemmDistanceL2,
        // Radians under ANGULAR, unit-vector distance under L2, as in sift.cc.
        static_cast<float>(options_.sift->max_distance),
        static_cast<float>(kSqSiftDescriptorNorm));

    // Keep the first max_num_matches in image1 order, as SiftGPU does
    // (SiftMatchCU::GetBestMatch); raw_matches is already in that order.
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
    LOG(FATAL_THROW) << "Guided matching not supported for CUSIFT_GEMM.";
  }

 private:
  const FeatureMatchingOptions options_;
  // cuBLAS handle and device buffers, freed when the matcher is destroyed.
  GemmMatcher gemm_;
};

#endif  // COLMAP_CUSIFT_ENABLED

}  // namespace

bool CusiftMatchingOptions::Check() const { return true; }

std::unique_ptr<FeatureMatcher> CreateCusiftFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_CUSIFT_ENABLED
  LOG(INFO) << "Creating cusift GEMM feature matcher";
  return CusiftFeatureMatcher::Create(options);
#else
  LOG(ERROR) << "COLMAP was built without cusift (CUSIFT_ENABLED, which needs "
                "CUDA >= 11.2)";
  return nullptr;
#endif  // COLMAP_CUSIFT_ENABLED
}

}  // namespace colmap
