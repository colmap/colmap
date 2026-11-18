// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/cusift.h"
#include "colmap/feature/sift.h"
#include "colmap/util/logging.h"

#ifdef COLMAP_CUSIFT_ENABLED
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <limits>
#include <vector>

#ifdef COLMAP_CUSIFT_ENABLED
// No CUDA headers: colmap_feature takes no CUDA runtime dependency.
#include "cusift_device.h"
#include "matcher_gemm.h"
#endif  // COLMAP_CUSIFT_ENABLED

namespace colmap {
namespace {

#ifdef COLMAP_CUSIFT_ENABLED

// COLMAP's one-way SIFT acceptance (sift.h), applied to the GEMM's top-2
// reduction keys of one query descriptor.
struct GemmTop2Acceptance {
  SiftMatchAcceptance sift;
  bool angular = true;

  bool Accepts(const GemmTop2& top2, const int k) const {
    if (angular) {
      // The brute-force reference seeds the runner-up at dot 0, so a missing
      // or non-positive runner-up counts as dot 0.
      return sift.AcceptsAngular(top2.best[k], std::max(top2.second[k], 0.0f));
    }
    // A missing runner-up is at FLT_MAX, as in the index matcher.
    constexpr float kMissing = std::numeric_limits<float>::max();
    const float second_sq_dist =
        top2.second[k] == -kMissing
            ? kMissing
            : SqDistanceFromGemmKey(top2.sqNorm[k], top2.second[k]);
    return sift.AcceptsL2(SqDistanceFromGemmKey(top2.sqNorm[k], top2.best[k]),
                          second_sq_dist);
  }
};

// Uploads COLMAP's row-major uint8 descriptors as they are, without per-row
// normalization: the reference (FindBestMatchesOneWayBruteForce) ranks on the
// raw inner product with a fixed 512^2 norm, and normalizing would re-rank
// near-ties.

class CusiftFeatureMatcher : public FeatureMatcher {
 public:
  // `device` is the GPU Create bound; Match binds its calling thread to it
  // again (the cuBLAS handle and the device buffers live there).
  CusiftFeatureMatcher(const FeatureMatchingOptions& options, const int device)
      : options_(options),
        device_(device),
        acceptance_(CheckedAcceptance(options)) {}

  static std::unique_ptr<FeatureMatcher> Create(
      const FeatureMatchingOptions& options) {
    BindCusiftDevice(options.gpu_index);
    auto matcher = std::make_unique<CusiftFeatureMatcher>(
        options, cusift::CurrentDevice());
    // Size the device buffers once when the worker has clamped
    // max_num_matches to the largest keypoint count
    // (feature_matching_utils.cc): growing them later calls cudaFree, which
    // synchronises the whole device.
    if (options.cusift->reserve_max_num_matches) {
      matcher->gemm_.Reserve(options.max_num_matches);
    }
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
    THROW_CHECK_EQ(image1.descriptors->data.cols(), kSiftDescriptorDim);
    THROW_CHECK_EQ(image2.descriptors->data.cols(), kSiftDescriptorDim);

    SetBestCudaDevice(device_);
    matches->clear();
    const int n1 = static_cast<int>(image1.descriptors->data.rows());
    const int n2 = static_cast<int>(image2.descriptors->data.rows());
    if (n1 == 0 || n2 == 0) {
      return;
    }

    gemm_.Top2(image1.descriptors->data.data(),
               n1,
               image2.descriptors->data.data(),
               n2,
               acceptance_.angular ? kGemmDistanceAngular : kGemmDistanceL2,
               options_.sift->cross_check,
               rows_,
               cols_);
    AcceptMatches(matches);
  }

  void MatchGuided(double /*max_error*/,
                   const Image& /*image1*/,
                   const Image& /*image2*/,
                   TwoViewGeometry* /*two_view_geometry*/) override {
    LOG(FATAL_THROW) << "Guided matching not supported for CUSIFT_GEMM.";
  }

 private:
  static GemmTop2Acceptance CheckedAcceptance(
      const FeatureMatchingOptions& options) {
    // Includes use_gpu and guided_matching, for callers that skip Check().
    THROW_CHECK(options.Check())
        << "invalid options for the cusift matcher; the reason is logged above";
    return {{static_cast<float>(options.sift->max_ratio),
             static_cast<float>(options.sift->max_distance)},
            options.cusift->distance_type == CusiftDistanceType::ANGULAR};
  }

  // Keeps each image1 descriptor's best candidate that passes the one-way test
  // and, with cross_check, is the best of its candidate in turn (and, with
  // cross_check_reverse_ratio, passes the one-way test in that direction too).
  // Stops after max_num_matches in image1 order, as SiftGPU does
  // (SiftMatchCU::GetBestMatch).
  void AcceptMatches(FeatureMatches* matches) const {
    const bool cross_check = options_.sift->cross_check;
    const bool reverse_test = options_.cusift->cross_check_reverse_ratio;
    const size_t max_num_matches =
        static_cast<size_t>(options_.max_num_matches);
    for (int i = 0; i < static_cast<int>(rows_.bestIdx.size()) &&
                    matches->size() < max_num_matches;
         ++i) {
      const int j = rows_.bestIdx[i];
      if (j < 0 || !acceptance_.Accepts(rows_, i)) {
        continue;
      }
      if (cross_check && (cols_.bestIdx[j] != i ||
                          (reverse_test && !acceptance_.Accepts(cols_, j)))) {
        continue;
      }
      matches->emplace_back(static_cast<point2D_t>(i),
                            static_cast<point2D_t>(j));
    }
  }

  const FeatureMatchingOptions options_;
  const int device_;
  const GemmTop2Acceptance acceptance_;
  // cuBLAS handle and device buffers, freed when the matcher is destroyed.
  GemmMatcher gemm_;
  // Top-2 results of the last pair, kept to reuse their allocations.
  GemmTop2 rows_;
  GemmTop2 cols_;
};

#endif  // COLMAP_CUSIFT_ENABLED

}  // namespace

bool CusiftMatchingOptions::Check() const {
  CHECK_OPTION(distance_type == CusiftDistanceType::L2 ||
               distance_type == CusiftDistanceType::ANGULAR);
  return true;
}

float SqDistanceFromGemmKey(const float sq_norm, const float key) {
  return std::max(0.0f, sq_norm - key);
}

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
