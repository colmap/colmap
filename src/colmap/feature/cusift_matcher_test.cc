// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/index.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/sift_test_utils.h"
#include "colmap/feature/utils.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/util/cache.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#if defined(COLMAP_CUSIFT_ENABLED)
#include "matcher_gemm.h"
#endif

namespace colmap {
namespace {

FeatureMatchingOptions CreateCusiftMatchingOptions() {
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  return options;
}

TEST(CusiftMatchingOptions, DefaultsMatchTheDefaultCpuMatcherRule) {
  // Bidirectional ratio test on L2 distances, as FindBestMatchesIndex.
  const CusiftMatchingOptions options;
  EXPECT_TRUE(options.cross_check_reverse_ratio);
  EXPECT_EQ(options.distance_type, CusiftDistanceType::L2);
}

TEST(CusiftMatchingOptions, RejectsUnknownDistanceType) {
  CusiftMatchingOptions options;
  EXPECT_TRUE(options.Check());
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.distance_type = static_cast<CusiftDistanceType>(7);
  EXPECT_FALSE(options.Check());
}

TEST(SqDistanceFromGemmKey, RecoversTheSquaredDistance) {
  // |q|^2 = 25 and key = 2 q.c - |c|^2 = 9 give |q - c|^2 = 16.
  EXPECT_EQ(SqDistanceFromGemmKey(25.0f, 9.0f), 16.0f);
}

TEST(SqDistanceFromGemmKey, ClampsRoundingBelowZeroToZero) {
  // With reduced-precision math, near-equal descriptors could round below 0,
  // which the ratio test's sqrt would turn into NaN.
  EXPECT_EQ(SqDistanceFromGemmKey(262144.0f, 262144.03f), 0.0f);
}

// Check() as CUSIFT_GEMM runs it. A build without cusift rejects the type
// before its SIFT checks, so there only the SIFT options' own Check() runs.
bool CheckAsCusiftGemm(const FeatureMatchingOptions& options) {
#if defined(COLMAP_CUSIFT_ENABLED)
  return options.Check();
#else
  return options.sift->Check();
#endif
}

TEST(CusiftMatchingOptions, ChecksSiftThresholds) {
  // The ratio and distance thresholds come from the SIFT options, and are
  // checked for CUSIFT_GEMM too.
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  EXPECT_TRUE(CheckAsCusiftGemm(options));
  options.sift->max_ratio = -1;
  EXPECT_FALSE(CheckAsCusiftGemm(options));
  options = CreateCusiftMatchingOptions();
  options.sift->max_distance = -1;
  EXPECT_FALSE(CheckAsCusiftGemm(options));
}

// Options pass Check() only in a build with cusift; without it, Check() rejects
// the cusift types before reaching cusift's own checks.
#if defined(COLMAP_CUSIFT_ENABLED)
TEST(CusiftFeatureMatcher, RequiresGpu) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  EXPECT_TRUE(options.Check());
  // cusift has no CPU implementation, so this must be rejected as a bad
  // option rather than surfacing later as an unknown-matcher-type error.
  options.use_gpu = false;
  EXPECT_FALSE(options.Check());
}
#endif  // COLMAP_CUSIFT_ENABLED

#if defined(COLMAP_CUSIFT_ENABLED)

const Camera& TestCamera() {
  static const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  return camera;
}

FeatureMatcher::Image MakeImage(image_t id, const FeatureDescriptors& desc) {
  return FeatureMatcher::Image{id,
                               &TestCamera(),
                               /*keypoints=*/nullptr,
                               std::make_shared<FeatureDescriptors>(desc)};
}

// Both cross-check modes run the same assertions where behaviour must agree;
// where they differ is asserted in
// ReverseRatioRejectsAmbiguousReverseDirection.
class CusiftCrossCheckModeTest : public ::testing::TestWithParam<bool> {};

TEST_P(CusiftCrossCheckModeTest, SelfMatchIdentity) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check_reverse_ratio = GetParam();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  FeatureDescriptors desc = CreateRandomFeatureDescriptors(16);
  const FeatureMatcher::Image image1 = MakeImage(1, desc);
  const FeatureMatcher::Image image2 = MakeImage(2, desc);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  // Each descriptor's unique nearest neighbour is itself, in either mode.
  EXPECT_GT(matches.size(), 0);
  for (const auto& m : matches) {
    EXPECT_EQ(m.point2D_idx1, m.point2D_idx2);
  }
}

TEST_P(CusiftCrossCheckModeTest, EmptyAndReversedInputs) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check_reverse_ratio = GetParam();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  const FeatureMatcher::Image image0 = MakeImage(0, CreateEmptyDescriptors());
  const FeatureMatcher::Image image1 =
      MakeImage(1, CreateRandomFeatureDescriptors(2));
  const FeatureMatcher::Image image2 =
      MakeImage(2, CreateReversedDescriptors(*image1.descriptors));

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);
  // Again, through the thread-local context's buffer-reuse path.
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

INSTANTIATE_TEST_SUITE_P(CusiftCrossCheckMode,
                         CusiftCrossCheckModeTest,
                         ::testing::Values(true, false),
                         [](const auto& info) {
                           return info.param ? "ReverseRatio" : "MutualOnly";
                         });

TEST(CusiftFeatureMatcher, RejectsGuidedMatching) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.guided_matching = true;
  // Rejected as a bad option, before the guided pass constructs the matcher on
  // a worker thread, and again by the constructor for callers that skip it.
  EXPECT_FALSE(options.Check());
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  EXPECT_THAT([&]() { CreateCusiftFeatureMatcher(options); },
              testing::ThrowsMessage<std::invalid_argument>(
                  testing::HasSubstr("invalid options for the cusift")));
  // And a matcher created for unguided matching refuses a guided pass.
  options.guided_matching = false;
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
  TwoViewGeometry two_view_geometry;
  EXPECT_THAT(
      [&]() {
        matcher->MatchGuided(/*max_error=*/4.0,
                             FeatureMatcher::Image{},
                             FeatureMatcher::Image{},
                             &two_view_geometry);
      },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::HasSubstr("Guided matching not supported")));
}

TEST(CusiftFeatureMatcher, ConstructorRejectsInvalidOptions) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Callers that skip FeatureMatchingOptions::Check(), such as a direct
  // Create, must still be stopped by the constructor.
  const auto expect_rejected = [](const FeatureMatchingOptions& options) {
    EXPECT_THAT([&]() { CreateCusiftFeatureMatcher(options); },
                testing::ThrowsMessage<std::invalid_argument>(
                    testing::HasSubstr("invalid options for the cusift")));
  };
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.sift->max_ratio = 0.0;
  expect_rejected(options);
  options = CreateCusiftMatchingOptions();
  options.sift->max_distance = -1.0;
  expect_rejected(options);
  options = CreateCusiftMatchingOptions();
  options.use_gpu = false;
  expect_rejected(options);
}

TEST(CusiftFeatureMatcher, TypeMismatch) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = MakeImage(1, sift_desc);
  const FeatureMatcher::Image image_undefined = MakeImage(2, undefined_desc);

  FeatureMatches matches;
  EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
               std::invalid_argument);

  // SIFT-typed but 64 wide: rejected before the upload reads 128 bytes a row.
  const FeatureMatcher::Image image_narrow =
      MakeImage(3,
                FeatureDescriptors(FeatureExtractorType::SIFT,
                                   FeatureDescriptorsData::Zero(2, 64)));
  EXPECT_THROW(matcher->Match(image_sift, image_narrow, &matches),
               std::invalid_argument);
  EXPECT_THROW(matcher->Match(image_narrow, image_sift, &matches),
               std::invalid_argument);
}

TEST(CusiftFeatureMatcher, MaxDistanceFiltersCorrectly) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Two descriptors sharing 8 of their 10 support dimensions: about 0.64 apart
  // both in radians and in unit-vector L2, so one pair tests both units.
  FeatureDescriptorsFloatData data = FeatureDescriptorsFloatData::Zero(2, 128);
  for (int d = 0; d < 10; ++d) data(0, d) = 1.0f;
  for (int d = 2; d < 12; ++d) data(1, d) = 1.0f;
  L2NormalizeFeatureDescriptors(&data);
  const FeatureDescriptorsData bytes = FeatureDescriptorsToUnsignedByte(data);
  const Eigen::VectorXf a = bytes.row(0).cast<float>();
  const Eigen::VectorXf b = bytes.row(1).cast<float>();
  const FeatureMatcher::Image image1 = MakeImage(
      1, FeatureDescriptors(FeatureExtractorType::SIFT, bytes.row(0)));
  const FeatureMatcher::Image image2 = MakeImage(
      2, FeatureDescriptors(FeatureExtractorType::SIFT, bytes.row(1)));

  for (const auto distance_type :
       {CusiftDistanceType::ANGULAR, CusiftDistanceType::L2}) {
    // The pair's distance in the units max_distance has in this mode, from
    // the quantized descriptors and the fixed norm of 512, as sift.cc does.
    constexpr float kNorm = 512.0f;
    const double distance =
        distance_type == CusiftDistanceType::ANGULAR
            ? std::acos(std::min(a.dot(b) / (kNorm * kNorm), 1.0f))
            : (a - b).norm() / kNorm;
    ASSERT_GT(distance, 0.3);
    FeatureMatchingOptions options = CreateCusiftMatchingOptions();
    options.cusift->distance_type = distance_type;
    // A single candidate's runner-up counts as infinitely far (L2) or at
    // angle pi/2 (ANGULAR); the pair's distance passes either ratio test.
    options.sift->max_ratio = 1.0;
    // Cross-checking must not change the outcome: the lone reverse candidate
    // has no runner-up either.
    for (const bool cross_check : {false, true}) {
      options.sift->cross_check = cross_check;
      for (const double factor : {1.05, 0.95}) {
        options.sift->max_distance = factor * distance;
        auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
        FeatureMatches matches;
        matcher->Match(image1, image2, &matches);
        EXPECT_EQ(matches.size(), factor > 1 ? 1 : 0)
            << "distance_type=" << distance_type << " distance=" << distance
            << " max_distance=" << options.sift->max_distance
            << " cross_check=" << cross_check;
      }
    }
  }
}

// Builds a descriptor whose uint8 norm deviates from COLMAP's nominal 512, by
// scaling a unit vector before quantizing. Uniform-norm test data hides an
// entire class of bug: when every candidate has the same norm, ranking by raw
// inner product and ranking by normalized distance are the same ordering, so a
// matcher that normalizes per row looks correct. Real descriptors do NOT have
// uniform norms.
FeatureDescriptors CreateDescriptorsWithVaryingNorms(const size_t num_features,
                                                     const int seed = 7) {
  SetPRNGSeed(seed);
  FeatureDescriptorsFloatData f =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    // Vary both the support size and the amplitude so the resulting uint8 rows
    // land at visibly different norms.
    const size_t support = 6 + (i % 11);
    const float amp = 0.55f + 0.45f * static_cast<float>(i % 5) / 4.0f;
    for (size_t j = 0; j < support; ++j) f(i, dims[j]) = amp;
    f.row(i) /= f.row(i).norm();
    f.row(i) *= (0.75f + 0.25f * static_cast<float>((i * 7) % 9) / 8.0f);
  }
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsToUnsignedByte(f));
}

std::set<std::pair<point2D_t, point2D_t>> MatchToSet(
    FeatureMatcher& matcher,
    const FeatureMatcher::Image& a,
    const FeatureMatcher::Image& b) {
  FeatureMatches m;
  matcher.Match(a, b, &m);
  std::set<std::pair<point2D_t, point2D_t>> out;
  for (const auto& x : m) out.emplace(x.point2D_idx1, x.point2D_idx2);
  return out;
}

TEST(CusiftFeatureMatcher, ReverseRatioRejectsAmbiguousReverseDirection) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Two near-duplicate
  // descriptors in image1 and one descriptor in image2 close to both:
  //   forward  - each image1 row has a single candidate, so no runner-up exists
  //              and the ratio test cannot reject it;
  //   reverse  - image2's row has two nearly equidistant candidates, so its
  //              ratio test MUST reject.
  // Mutual-best agreement alone therefore keeps a match that the reference
  // rejects. Asserting only `strict subset-of loose` passes vacuously when the
  // two modes agree, which they do on near-orthogonal random data -- so assert
  // the strict difference on data built to discriminate.
  FeatureDescriptorsFloatData f = FeatureDescriptorsFloatData::Zero(2, 128);
  for (int d = 0; d < 10; ++d) f(0, d) = 1.0f;
  for (int d = 0; d < 10; ++d) f(1, d) = 1.0f;
  f(1, 10) = 0.06f;  // b = a + tiny perturbation
  f.row(0) /= f.row(0).norm();
  f.row(1) /= f.row(1).norm();
  const FeatureDescriptors d1(FeatureExtractorType::SIFT,
                              FeatureDescriptorsToUnsignedByte(f));

  FeatureDescriptorsFloatData g = FeatureDescriptorsFloatData::Zero(1, 128);
  for (int d = 0; d < 10; ++d) g(0, d) = 1.0f;
  g(0, 10) = 0.03f;  // c sits between a and b
  g.row(0) /= g.row(0).norm();
  const FeatureDescriptors d2(FeatureExtractorType::SIFT,
                              FeatureDescriptorsToUnsignedByte(g));

  const auto img1 = MakeImage(1, d1);
  const auto img2 = MakeImage(2, d2);

  auto run = [&](bool reverse_ratio) {
    FeatureMatchingOptions options = CreateCusiftMatchingOptions();
    options.cusift->cross_check_reverse_ratio = reverse_ratio;
    auto m = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
    return MatchToSet(*m, img1, img2);
  };
  const auto strict = run(/*reverse_ratio=*/true);
  const auto loose = run(/*reverse_ratio=*/false);

  EXPECT_TRUE(strict.empty())
      << "reverse-direction ratio test should reject the ambiguous match";
  EXPECT_EQ(loose.size(), 1)
      << "mutual-best-only should keep it -- if this fails the fixture no "
         "longer discriminates between the two modes and must be rebuilt";
}

// CANONICAL TIE-BREAK, column direction.
//
// Only the column side is testable through the public API, and it is worth
// recording why the row side is not: the epilogue reads the row runner-up's
// VALUE (hS) and uses its index only as an exists/not-exists sentinel, so a
// second-slot tie is invisible; and a best-slot tie makes best == second, which
// the ratio test rejects for any ratio <= 1 (the comparison is `>=`), so it is
// invisible too. A "row tie-break" test would therefore assert nothing.
//
// The column side IS observable because the cross-check compares hColIdx[j]
// against i for equality, so which of two tied rows wins decides which pair
// survives.
//
// The fixture must distinguish the lowest row index from the lowest thread
// index: ColTop2 block-strides rows over 256 threads, so a reduction that
// resolved ties by threadIdx.x would pick the lowest (r % 256). Rows 0 and 1
// map to threads 0 and 1, where both rules agree. Rows 45 and 300 map to
// threads 45 and 44, so a thread-order rule picks row 300 while the canonical
// rule picks row 45.
//
// reverse_ratio is disabled here on purpose: two tied candidates make the
// column's own best == second, which its ratio test would reject, removing the
// very match this test needs to observe.
TEST(CusiftFeatureMatcher, ColumnTieBreakPrefersLowerRowIndex) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  constexpr int kN1 = 320;       // > 300 so row 300 exists
  constexpr int kTieLow = 45;    // -> thread 45
  constexpr int kTieHigh = 300;  // -> thread 44, wins under a thread-order rule
  static_assert(
      kTieHigh % 256 < kTieLow % 256,
      "fixture must discriminate: the higher row must map to a lower "
      "thread, or both tie-break rules agree and the test is vacuous");

  FeatureDescriptors d1 = CreateRandomFeatureDescriptors(kN1);
  // Make the two rows byte-identical, so their integer dot products against any
  // query are exactly equal -- a real tie, not an approximate one.
  d1.data.row(kTieHigh) = d1.data.row(kTieLow);

  // image2 holds a single descriptor equal to the tied pair, so both are its
  // joint-best. One descriptor also means the ROW side has no runner-up, so the
  // row ratio test cannot reject the match before the cross-check is reached.
  FeatureDescriptorsData d2data(1, 128);
  d2data.row(0) = d1.data.row(kTieLow);
  const FeatureDescriptors d2(FeatureExtractorType::SIFT, d2data);

  const auto img1 = MakeImage(1, d1);
  const auto img2 = MakeImage(2, d2);

  // The key changes the compared VALUE between distance types but must not
  // change the tie rule.
  for (const auto dt : {CusiftDistanceType::ANGULAR, CusiftDistanceType::L2}) {
    FeatureMatchingOptions options = CreateCusiftMatchingOptions();
    options.cusift->distance_type = dt;
    options.sift->cross_check = true;
    options.cusift->cross_check_reverse_ratio = false;
    auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

    FeatureMatches matches;
    matcher->Match(img1, img2, &matches);
    ASSERT_EQ(matches.size(), 1) << "distance_type=" << static_cast<int>(dt);
    EXPECT_EQ(matches[0].point2D_idx1, static_cast<point2D_t>(kTieLow))
        << "column tie must resolve to the LOWER row index (" << kTieLow
        << "), not the one that happens to land on a lower thread (" << kTieHigh
        << "); distance_type=" << static_cast<int>(dt);
    EXPECT_EQ(matches[0].point2D_idx2, 0);
  }
}

// A second set with genuine correspondences: independent random sparse
// descriptors are near-orthogonal and produce almost no matches, so both
// matchers would return the empty set and an equality assertion would pass
// vacuously. Perturbed copies give real matches AND keep the near-tie
// population that makes a differential test sensitive. The perturbation adds
// 6 to 10 to one dimension per row; the L2 thresholds of
// DifferentialAcrossTilesMetricsAndCrossCheckModes are chosen around it.
FeatureDescriptors CreatePerturbedCopy(const FeatureDescriptors& src,
                                       const int seed) {
  FeatureDescriptorsFloatData f = src.ToFloat().data;
  for (Eigen::Index r = 0; r < f.rows(); ++r) {
    const int dim = static_cast<int>((r * 37 + seed) % 128);
    f(r, dim) += 6.0f + static_cast<float>((r + seed) % 5);
  }
  return FeatureDescriptors(FeatureExtractorType::SIFT, f.cast<uint8_t>());
}

// The exact reference implements only ANGULAR, so these parity tests set it.
FeatureMatchingOptions CreateCusiftAngularMatchingOptions() {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->distance_type = CusiftDistanceType::ANGULAR;
  return options;
}

using DescriptorIndexCache =
    ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>;

// The descriptor indexes COLMAP's index-based CPU matcher reads; its matching
// controller builds them the same way.
std::unique_ptr<DescriptorIndexCache> CreateDescriptorIndexCache(
    const FeatureMatcher::Image& image1, const FeatureMatcher::Image& image2) {
  return std::make_unique<DescriptorIndexCache>(
      2, [&image1, &image2](const image_t image_id) {
        const auto& image = image_id == image1.image_id ? image1 : image2;
        auto index = FeatureDescriptorIndex::Create();
        index->Build(image.descriptors->ToFloat());
        return index;
      });
}

// Differential fuzz against COLMAP's CPU SIFT matchers, on descriptors with
// NON-UNIFORM norms, where ranking by raw inner product and ranking by
// normalized distance disagree. Each pair is matched by cusift and by the CPU
// `reference` (on cusift's thresholds), and the match sets must be identical:
// bit-exactness is the ideal property to fuzz, with no tolerance to tune.
// Returns the number of reference matches, for the non-vacuity guards.
size_t ExpectSameMatchesOnFuzzPairs(const FeatureMatchingOptions& cusift,
                                    FeatureMatchingOptions reference,
                                    const std::vector<size_t>& sizes) {
  reference.use_gpu = false;
  reference.sift->max_ratio = cusift.sift->max_ratio;
  reference.sift->max_distance = cusift.sift->max_distance;
  reference.sift->cross_check = cusift.sift->cross_check;
  size_t total_reference_matches = 0;
  for (int seed = 1; seed <= 12; ++seed) {
    for (const size_t n : sizes) {
      const FeatureDescriptors da = CreateDescriptorsWithVaryingNorms(n, seed);
      const auto img1 = MakeImage(1, da);
      const auto img2 = MakeImage(2, CreatePerturbedCopy(da, seed));
      const auto index_cache = CreateDescriptorIndexCache(img1, img2);
      reference.sift->cpu_descriptor_index_cache = index_cache.get();
      const auto got = MatchToSet(
          *THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cusift)), img1, img2);
      const auto want =
          MatchToSet(*THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(reference)),
                     img1,
                     img2);
      total_reference_matches += want.size();
      EXPECT_EQ(got, want) << "cusift diverged from the CPU reference at n="
                           << n << " seed=" << seed << " (got " << got.size()
                           << " vs " << want.size() << " matches)";
    }
  }
  return total_reference_matches;
}

// The reference is `cpu_brute_force_matcher = true`, COLMAP's exact matcher,
// which implements only ANGULAR.
TEST(CusiftMatcherParity, DifferentialAgainstExactCpuBruteForce) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions reference(FeatureMatcherType::SIFT_BRUTEFORCE);
  reference.sift->cpu_brute_force_matcher = true;
  // Non-vacuity guard: if the fixtures stop producing matches this test
  // silently becomes "empty == empty" and stops testing anything at all.
  EXPECT_GT(ExpectSameMatchesOnFuzzPairs(CreateCusiftAngularMatchingOptions(),
                                         reference,
                                         {2u, 3u, 17u, 64u, 200u}),
            100u)
      << "fixtures produced too few reference matches to be meaningful";
}

// Both at their defaults: cusift's L2 rule and the CPU's default faiss index
// matcher. faiss searches exactly (IndexFlatL2) below 512 indexed descriptors
// and approximately (IVF, 8 probed cells) from 512, so the match sets can be
// required equal only below that size.
TEST(CusiftMatcherParity, DefaultMatchesDefaultCpuMatcherBelowIvfSize) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions reference(FeatureMatcherType::SIFT_BRUTEFORCE);
  ASSERT_FALSE(reference.sift->cpu_brute_force_matcher);
  EXPECT_GT(ExpectSameMatchesOnFuzzPairs(CreateCusiftMatchingOptions(),
                                         reference,
                                         {2u, 3u, 17u, 64u, 200u, 511u}),
            100u)
      << "fixtures produced too few reference matches to be meaningful";
}

// NEGATIVE CONTROL. Every other test here matches RELATED content, so they
// exercise only the acceptance path. The ratio test and cross-check exist to
// SUPPRESS matches between unrelated content, and a regression there is
// invisible to positive-pair tests by construction.
TEST(CusiftMatcherParity, NegativeControlUnrelatedDescriptors) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Independently generated sets: nothing should legitimately correspond.
  const FeatureDescriptors da = CreateDescriptorsWithVaryingNorms(150, 21);
  const FeatureDescriptors db = CreateDescriptorsWithVaryingNorms(150, 99);
  const auto img1 = MakeImage(1, da);
  const auto img2 = MakeImage(2, db);

  FeatureMatchingOptions cu = CreateCusiftAngularMatchingOptions();
  auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));
  FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
  ref.use_gpu = false;
  ref.sift->cpu_brute_force_matcher = true;
  auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

  const auto got = MatchToSet(*cu_matcher, img1, img2);
  const auto want = MatchToSet(*ref_matcher, img1, img2);
  // Parity alone would pass if both matchers stopped suppressing.
  EXPECT_TRUE(want.empty()) << want.size() << " reference matches";
  EXPECT_TRUE(got.empty()) << got.size() << " cusift matches";
  EXPECT_EQ(got, want);
}

// Descriptors whose norms deliberately miss COLMAP's nominal
// 512 (here ~400 and ~600), placed so that ranking by RAW inner product and
// ranking by NORMALIZED distance choose different winners. Under per-row
// normalization the decision flips; under the reference's fixed-512 convention
// it does not. Same boundary-test discipline as MaxDistanceFiltersCorrectly,
// applied to the ratio/cross-check semantics rather than only to max_distance.
TEST(CusiftMatcherParity, NonCanonicalNormsDoNotChangeTheDecision) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";

  FeatureDescriptorsFloatData q = FeatureDescriptorsFloatData::Zero(1, 128);
  for (int d = 0; d < 12; ++d) q(0, d) = 1.0f;
  q.row(0) /= q.row(0).norm();

  // c0: shorter (norm ~400/512) but better aligned with q.
  // c1: longer  (norm ~600/512) but slightly less aligned.
  // Raw dot can favour c1 while normalized cosine favours c0.
  FeatureDescriptorsFloatData c = FeatureDescriptorsFloatData::Zero(2, 128);
  for (int d = 0; d < 12; ++d) c(0, d) = 1.0f;
  c.row(0) /= c.row(0).norm();
  c.row(0) *= (400.0f / 512.0f);
  for (int d = 0; d < 12; ++d) c(1, d) = 1.0f;
  c(1, 40) = 0.42f;  // pull it off q's direction
  c.row(1) /= c.row(1).norm();
  c.row(1) *= (600.0f / 512.0f);

  const FeatureDescriptors dq(FeatureExtractorType::SIFT,
                              FeatureDescriptorsToUnsignedByte(q));
  const FeatureDescriptors dc(FeatureExtractorType::SIFT,
                              FeatureDescriptorsToUnsignedByte(c));
  const auto img1 = MakeImage(1, dq);
  const auto img2 = MakeImage(2, dc);

  FeatureMatchingOptions cu = CreateCusiftAngularMatchingOptions();
  cu.sift->max_ratio = 0.99;  // keep the ratio test from masking the choice
  auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));
  FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
  ref.use_gpu = false;
  ref.sift->cpu_brute_force_matcher = true;
  ref.sift->max_ratio = 0.99;
  auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

  const auto want = MatchToSet(*ref_matcher, img1, img2);
  ASSERT_EQ(want.size(), 1);
  EXPECT_EQ(MatchToSet(*cu_matcher, img1, img2), want)
      << "the winning candidate changed under non-canonical descriptor norms";
}

// Thresholds and cross-check settings, applied alike to the cusift matcher
// and to the in-test reference; COLMAP's exact CPU matcher implements only
// ANGULAR with the reverse ratio test.
struct MatchingSetup {
  double max_ratio = 0.8;
  double max_distance = 0.7;
  bool cross_check = true;
  CusiftDistanceType distance_type = CusiftDistanceType::ANGULAR;
  bool cross_check_reverse_ratio = true;
};

FeatureMatchingOptions CreateCusiftMatchingOptions(const MatchingSetup& setup) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->distance_type = setup.distance_type;
  options.cusift->cross_check_reverse_ratio = setup.cross_check_reverse_ratio;
  options.sift->max_ratio = setup.max_ratio;
  options.sift->max_distance = setup.max_distance;
  options.sift->cross_check = setup.cross_check;
  return options;
}

FeatureMatchingOptions CreateExactCpuMatchingOptions(
    const MatchingSetup& setup) {
  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  options.sift->max_ratio = setup.max_ratio;
  options.sift->max_distance = setup.max_distance;
  options.sift->cross_check = setup.cross_check;
  return options;
}

// Exact reference for both metrics and both cross-check strengths, written
// from the definitions in sift.cc rather than reusing its matchers, which
// implement only ANGULAR with the full reverse ratio test. One way, the best
// candidate must lie within max_distance, and the match is rejected once its
// distance reaches max_ratio times the second best's; ties go to the lower
// index. ANGULAR ranks on the raw inner product with the fixed 512^2 norm,
// and a lone candidate's runner-up counts as dot 0 (angle pi/2). L2 ranks on
// the squared distance, and a lone candidate's runner-up is infinitely far.
// Per row of `query`, the index of its accepted candidate in `target`, or -1.
// Without `thresholds` the best candidate is returned untested: the reverse
// direction of a mutual-best-only cross-check.
std::vector<int> OracleOneWay(const Eigen::MatrixXf& query,
                              const Eigen::MatrixXf& target,
                              const MatchingSetup& setup,
                              const bool thresholds) {
  constexpr float kSqNorm = 512.0f * 512.0f;
  // Static, so the capture-less lambda below may use it (MSVC requires that).
  static constexpr float kInvSqNorm = 1.0f / kSqNorm;
  constexpr float kMissing = std::numeric_limits<float>::max();
  const auto angle = [](const float dot) {
    return std::acos(std::min(kInvSqNorm * dot, 1.0f));
  };
  const bool angular = setup.distance_type == CusiftDistanceType::ANGULAR;
  const float max_ratio = static_cast<float>(setup.max_ratio);
  const float max_distance = static_cast<float>(setup.max_distance);
  const float max_sq_dist = kSqNorm * max_distance * max_distance;

  // Sums of uint8 products stay below 2^24, so every value here is exact.
  const Eigen::MatrixXf dots = query * target.transpose();
  const Eigen::VectorXf query_sq = query.rowwise().squaredNorm();
  const Eigen::VectorXf target_sq = target.rowwise().squaredNorm();
  std::vector<int> out(query.rows(), -1);
  for (Eigen::Index i = 0; i < query.rows(); ++i) {
    // A larger score is better: the dot, or the negated squared distance.
    float best = -kMissing;
    float second = -kMissing;
    int best_idx = -1;
    for (Eigen::Index j = 0; j < target.rows(); ++j) {
      const float score =
          angular ? dots(i, j)
                  : -(query_sq(i) + target_sq(j) - 2.0f * dots(i, j));
      if (score > best) {
        second = best;
        best = score;
        best_idx = static_cast<int>(j);
      } else if (score > second) {
        second = score;
      }
    }
    if (!thresholds) {
      out[i] = best_idx;
      continue;
    }
    if (angular) {
      if (best <= 0.0f || angle(best) > max_distance ||
          angle(best) >= max_ratio * angle(std::max(second, 0.0f))) {
        continue;
      }
    } else {
      const float second_sq_dist = second == -kMissing ? kMissing : -second;
      if (-best > max_sq_dist ||
          std::sqrt(-best) >= max_ratio * std::sqrt(second_sq_dist)) {
        continue;
      }
    }
    out[i] = best_idx;
  }
  return out;
}

std::set<std::pair<point2D_t, point2D_t>> OracleMatches(
    const FeatureDescriptors& da,
    const FeatureDescriptors& db,
    const MatchingSetup& setup) {
  const Eigen::MatrixXf a = da.data.cast<float>();
  const Eigen::MatrixXf b = db.data.cast<float>();
  const std::vector<int> forward =
      OracleOneWay(a, b, setup, /*thresholds=*/true);
  std::vector<int> reverse;
  if (setup.cross_check) {
    reverse = OracleOneWay(b, a, setup, setup.cross_check_reverse_ratio);
  }
  std::set<std::pair<point2D_t, point2D_t>> out;
  for (size_t i = 0; i < forward.size(); ++i) {
    const int j = forward[i];
    if (j < 0 || (setup.cross_check && reverse[j] != static_cast<int>(i))) {
      continue;
    }
    out.emplace(i, j);
  }
  return out;
}

FeatureDescriptors HeadRows(const FeatureDescriptors& src,
                            const Eigen::Index num_rows) {
  return FeatureDescriptors(src.type, src.data.topRows(num_rows));
}

FeatureDescriptors TailRows(const FeatureDescriptors& src,
                            const Eigen::Index num_rows) {
  return FeatureDescriptors(src.type, src.data.bottomRows(num_rows));
}

FeatureDescriptors ConcatRows(const FeatureDescriptors& top,
                              const FeatureDescriptors& bottom) {
  FeatureDescriptorsData data(top.data.rows() + bottom.data.rows(),
                              top.data.cols());
  data << top.data, bottom.data;
  return FeatureDescriptors(top.type, data);
}

// Zeroes the first nonzero dimension of each row. Unlike CreatePerturbedCopy,
// this moves every row away from its source under both metrics: the angular
// reference ranks on the raw dot product, which adding to a dimension can only
// keep or raise.
FeatureDescriptors CreateCopyMissingOneDimension(
    const FeatureDescriptors& src) {
  FeatureDescriptors copy = src;
  for (Eigen::Index r = 0; r < copy.data.rows(); ++r) {
    Eigen::Index dim = 0;
    while (dim < copy.data.cols() && copy.data(r, dim) == 0) ++dim;
    THROW_CHECK_LT(dim, copy.data.cols());
    copy.data(r, dim) = 0;
  }
  return copy;
}

// Image 2 holds degraded copies of the first kMaxNumMatches rows and exact
// copies of the rest, so the earliest rows match worst. A distance-sorted
// top-K would keep exact copies; the index-order rule (cusift_matcher.h) keeps
// the first rows.
TEST(CusiftFeatureMatcher, MaxNumMatchesTruncatesByIndexOrder) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  constexpr Eigen::Index kNumFeatures = 20;
  constexpr int kMaxNumMatches = 5;
  const FeatureDescriptors desc = CreateRandomFeatureDescriptors(kNumFeatures);
  const FeatureDescriptors desc2 =
      ConcatRows(CreateCopyMissingOneDimension(HeadRows(desc, kMaxNumMatches)),
                 TailRows(desc, kNumFeatures - kMaxNumMatches));
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.sift->cross_check = false;
  options.max_num_matches = kMaxNumMatches;
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  FeatureMatches matches;
  matcher->Match(MakeImage(1, desc), MakeImage(2, desc2), &matches);

  ASSERT_EQ(matches.size(), static_cast<size_t>(kMaxNumMatches));
  for (int i = 0; i < kMaxNumMatches; ++i) {
    EXPECT_EQ(matches[i].point2D_idx1, static_cast<point2D_t>(i));
    EXPECT_EQ(matches[i].point2D_idx2, static_cast<point2D_t>(i));
  }
}

// Sizes of the tiled differential fixture: both images hold the
// `num_base` descriptors, image 1 as is and image 2 perturbed. Image 1 appends
// perturbed near duplicates of the first base rows, which let two queries
// choose the same target, so only a cross-check removes one. Image 2 appends
// near duplicates of the last base rows, which give both ratio tests a close
// runner-up. Each copy has its own seed, so its perturbations land on other
// dimensions with other sizes.
struct TiledFixtureSize {
  size_t num_base;
  Eigen::Index num_head_dups1;
  Eigen::Index num_tail_dups2;
};

std::pair<FeatureDescriptors, FeatureDescriptors> CreateTiledFixture(
    const TiledFixtureSize& size, const int seed) {
  const FeatureDescriptors base =
      CreateDescriptorsWithVaryingNorms(size.num_base, seed);
  return {ConcatRows(base,
                     CreatePerturbedCopy(HeadRows(base, size.num_head_dups1),
                                         seed + 101)),
          ConcatRows(CreatePerturbedCopy(base, seed),
                     CreatePerturbedCopy(TailRows(base, size.num_tail_dups2),
                                         seed + 202))};
}

// Guards against a vacuous comparison, so a matcher that ignored a setting
// would fail: every max_distance step must admit more matches in each
// cross-check mode, and at the loosest max_distance each stronger mode must
// reject more. Tighter thresholds admit only unambiguous matches, which every
// mode keeps. Indexed [max_distance][mode], with max_distance ascending and
// the modes ordered none, mutual best, reverse ratio test.
void ExpectEveryAxisMatters(const std::vector<std::vector<size_t>>& num_matches,
                            const std::string& group) {
  for (size_t d = 1; d < num_matches.size(); ++d) {
    for (size_t m = 0; m < num_matches[d].size(); ++m) {
      EXPECT_LT(num_matches[d - 1][m], num_matches[d][m])
          << "max_distance has no effect at" << group;
    }
  }
  const std::vector<size_t>& loosest = num_matches.back();
  for (size_t m = 1; m < loosest.size(); ++m) {
    EXPECT_GT(loosest[m - 1], loosest[m])
        << "the cross-check mode has no effect at" << group;
  }
}

// The row reduction merges running top-2 results across 512-column tiles and
// the column reduction strides over more than 512 rows, so both sides exceed
// 512 in turn, never at equal sizes. L2 and the mutual-best-only cross-check
// have no CPU matcher to compare against, so the oracle is the reference for
// every combination; in the one configuration the exact CPU matcher
// implements, it pins the oracle itself.
TEST(CusiftMatcherParity, DifferentialAcrossTilesMetricsAndCrossCheckModes) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // n2 > 512 > n1; n1 > n2 > 512; n1 > 512 > n2.
  const std::vector<TiledFixtureSize> sizes = {
      {360, 120, 200}, {1000, 400, 100}, {450, 150, 30}};
  // L2 distances shrink with the perturbations, ANGULAR ones are dominated by
  // the descriptors' norm deficit; each list spans tight to loose.
  const std::vector<std::pair<CusiftDistanceType, std::vector<double>>>
      metrics = {{CusiftDistanceType::ANGULAR, {0.25, 0.7, 1.5}},
                 {CusiftDistanceType::L2, {0.015, 0.0185, 1.5}}};
  const std::vector<std::pair<bool, bool>> modes = {
      {false, true}, {true, false}, {true, true}};
  for (const TiledFixtureSize& size : sizes) {
    for (const int seed : {1, 2}) {
      const auto [da, db] = CreateTiledFixture(size, seed);
      const auto img1 = MakeImage(1, da);
      const auto img2 = MakeImage(2, db);
      for (const auto& [distance_type, max_distances] : metrics) {
        std::vector<std::vector<size_t>> num_matches;
        const std::string group =
            StringPrintf(" n1=%d n2=%d seed=%d distance_type=%d",
                         static_cast<int>(da.data.rows()),
                         static_cast<int>(db.data.rows()),
                         seed,
                         static_cast<int>(distance_type));
        for (const double max_distance : max_distances) {
          num_matches.emplace_back();
          for (const auto& [cross_check, reverse_ratio] : modes) {
            MatchingSetup setup;
            setup.max_distance = max_distance;
            setup.cross_check = cross_check;
            setup.distance_type = distance_type;
            setup.cross_check_reverse_ratio = reverse_ratio;
            const std::string config =
                group + StringPrintf(
                            " max_distance=%g cross_check=%d "
                            "reverse_ratio=%d",
                            max_distance,
                            cross_check,
                            reverse_ratio);
            const auto want = OracleMatches(da, db, setup);
            num_matches.back().push_back(want.size());
            auto cu_matcher = THROW_CHECK_NOTNULL(
                CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions(setup)));
            EXPECT_EQ(MatchToSet(*cu_matcher, img1, img2), want)
                << "cusift diverged from the oracle at" << config;
            if (distance_type == CusiftDistanceType::ANGULAR && cross_check &&
                reverse_ratio) {
              auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(
                  CreateExactCpuMatchingOptions(setup)));
              EXPECT_EQ(MatchToSet(*ref_matcher, img1, img2), want)
                  << "the oracle diverged from the exact CPU matcher at"
                  << config;
            }
          }
        }
        ExpectEveryAxisMatters(num_matches, group);
      }
    }
  }
}

// One descriptor in each image leaves both ratio tests without a runner-up.
// The reference seeds it at dot 0 (angle pi/2): a lone candidate at 1.37 rad
// fails ratio 0.8 even inside max_distance 1.5, one at 0.45 rad passes, and a
// candidate sharing no support (dot 0) is no candidate at all, however large
// max_distance is.
TEST(CusiftMatcherParity, SingleCandidateFollowsTheReference) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureDescriptorsFloatData f = FeatureDescriptorsFloatData::Zero(4, 128);
  for (int d = 0; d < 10; ++d) f(0, d) = 1.0f;  // the query
  for (int d = 8; d < 18; ++d) f(1, d) = 1.0f;  // shares 2 of 10: 1.37 rad
  for (int d = 0; d < 9; ++d) f(2, d) = 1.0f;   // shares 9 of 10: 0.45 rad
  f(2, 20) = 1.0f;
  for (int d = 30; d < 40; ++d) f(3, d) = 1.0f;  // disjoint: dot 0
  L2NormalizeFeatureDescriptors(&f);
  const FeatureDescriptorsData bytes = FeatureDescriptorsToUnsignedByte(f);
  const auto single = [&bytes](const int row) {
    return MakeImage(
        row + 1,
        FeatureDescriptors(FeatureExtractorType::SIFT, bytes.row(row)));
  };
  struct Case {
    int candidate;
    size_t num_matches;
  };
  for (const Case& c : {Case{1, 0}, Case{2, 1}, Case{3, 0}}) {
    for (const bool cross_check : {false, true}) {
      for (const double max_distance : {1.5, 1.6}) {
        const MatchingSetup setup{0.8, max_distance, cross_check};
        auto ref_matcher = THROW_CHECK_NOTNULL(
            CreateSiftFeatureMatcher(CreateExactCpuMatchingOptions(setup)));
        const auto want =
            MatchToSet(*ref_matcher, single(0), single(c.candidate));
        ASSERT_EQ(want.size(), c.num_matches);
        auto cu_matcher = THROW_CHECK_NOTNULL(
            CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions(setup)));
        EXPECT_EQ(MatchToSet(*cu_matcher, single(0), single(c.candidate)), want)
            << "candidate row " << c.candidate << " cross_check=" << cross_check
            << " max_distance=" << max_distance;
      }
    }
  }
  // With max_ratio above 1 only the no-candidate rule rejects the disjoint
  // candidate: its dot 0 would otherwise pass the ratio test against pi/2.
  const MatchingSetup lenient{/*max_ratio=*/1.2,
                              /*max_distance=*/1.6,
                              /*cross_check=*/false};
  auto ref_matcher = THROW_CHECK_NOTNULL(
      CreateSiftFeatureMatcher(CreateExactCpuMatchingOptions(lenient)));
  ASSERT_TRUE(MatchToSet(*ref_matcher, single(0), single(3)).empty());
  auto cu_matcher = THROW_CHECK_NOTNULL(
      CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions(lenient)));
  EXPECT_TRUE(MatchToSet(*cu_matcher, single(0), single(3)).empty());
}

TEST(SiftMatchAcceptance, AngularRejectsRunnerUpAsCloseAsBest) {
  const SiftMatchAcceptance acceptance{/*max_ratio=*/0.8f,
                                       /*max_distance=*/0.7f};
  EXPECT_FALSE(acceptance.AcceptsAngular(200000.0f, 200000.0f));
}

TEST(SiftMatchAcceptance, AngularRejectsNonPositiveBestDot) {
  const SiftMatchAcceptance acceptance{/*max_ratio=*/0.8f,
                                       /*max_distance=*/3.0f};
  EXPECT_FALSE(acceptance.AcceptsAngular(0.0f, 0.0f));
  EXPECT_FALSE(acceptance.AcceptsAngular(-1000.0f, 0.0f));
}

TEST(SiftMatchAcceptance, AngularTestsLoneCandidateAgainstRightAngle) {
  // A missing runner-up is passed as dot 0, i.e. an angle of pi/2.
  const float dot_at_one_radian = kSqSiftDescriptorNorm * std::cos(1.0f);
  const SiftMatchAcceptance passes_ratio{/*max_ratio=*/0.8f,
                                         /*max_distance=*/1.5f};
  EXPECT_TRUE(passes_ratio.AcceptsAngular(dot_at_one_radian, 0.0f));
  const SiftMatchAcceptance fails_ratio{/*max_ratio=*/0.6f,
                                        /*max_distance=*/1.5f};
  EXPECT_FALSE(fails_ratio.AcceptsAngular(dot_at_one_radian, 0.0f));
}

TEST(SiftMatchAcceptance, AngularRejectsBestBeyondMaxDistance) {
  const float dot_at_one_radian = kSqSiftDescriptorNorm * std::cos(1.0f);
  const SiftMatchAcceptance acceptance{/*max_ratio=*/0.8f,
                                       /*max_distance=*/0.7f};
  EXPECT_FALSE(acceptance.AcceptsAngular(dot_at_one_radian, 0.0f));
}

TEST(SiftMatchAcceptance, L2RejectsExactRatioBoundary) {
  // sqrt(16) == 0.8 * sqrt(25) in float: the strict test rejects it, as the
  // index matcher does.
  const SiftMatchAcceptance acceptance{/*max_ratio=*/0.8f,
                                       /*max_distance=*/0.7f};
  EXPECT_FALSE(acceptance.AcceptsL2(16.0f, 25.0f));
  EXPECT_TRUE(acceptance.AcceptsL2(16.0f, 26.0f));
}

TEST(SiftMatchAcceptance, L2AcceptsLoneCandidateWithinMaxDistance) {
  constexpr float kMissing = std::numeric_limits<float>::max();
  const SiftMatchAcceptance acceptance{/*max_ratio=*/0.8f,
                                       /*max_distance=*/0.7f};
  EXPECT_TRUE(acceptance.AcceptsL2(100.0f, kMissing));
  // 0.7^2 * 512^2 = 128450.56 is the largest accepted squared distance.
  EXPECT_FALSE(acceptance.AcceptsL2(130000.0f, kMissing));
}

// The vendored GEMM matcher's own contract, for callers that use it directly
// rather than through CusiftFeatureMatcher (which guards empty inputs).
TEST(GemmMatcherContract, EmptySideClearsAndReturnsNoCandidates) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const FeatureDescriptors descriptors = CreateRandomFeatureDescriptors(8);
  GemmMatcher matcher;
  for (const auto& [n1, n2] :
       std::vector<std::pair<int, int>>{{0, 8}, {8, 0}, {0, 0}}) {
    GemmTop2 rows;
    GemmTop2 cols;
    rows.Resize(3);
    cols.Resize(3);
    matcher.Top2(n1 > 0 ? descriptors.data.data() : nullptr,
                 n1,
                 n2 > 0 ? descriptors.data.data() : nullptr,
                 n2,
                 kGemmDistanceAngular,
                 /*columns=*/true,
                 rows,
                 cols);
    EXPECT_TRUE(rows.bestIdx.empty()) << "n1=" << n1 << " n2=" << n2;
    EXPECT_TRUE(cols.bestIdx.empty()) << "n1=" << n1 << " n2=" << n2;
  }
}

TEST(GemmMatcherContract, ReplacesPreviousCandidates) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const FeatureDescriptors descriptors = CreateRandomFeatureDescriptors(8);
  GemmMatcher matcher;
  GemmTop2 rows;
  GemmTop2 cols;
  for (int call = 0; call < 2; ++call) {
    matcher.Top2(descriptors.data.data(),
                 8,
                 descriptors.data.data(),
                 4,
                 kGemmDistanceAngular,
                 /*columns=*/call == 0,
                 rows,
                 cols);
  }
  ASSERT_EQ(rows.bestIdx.size(), 8);
  EXPECT_TRUE(cols.bestIdx.empty());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(rows.bestIdx[i], i);
  }
}

TEST(GemmMatcherContract, RejectsNullPointerForNonEmptySide) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const FeatureDescriptors descriptors = CreateRandomFeatureDescriptors(8);
  GemmMatcher matcher;
  GemmTop2 rows;
  GemmTop2 cols;
  EXPECT_THROW(matcher.Top2(nullptr,
                            8,
                            descriptors.data.data(),
                            8,
                            kGemmDistanceAngular,
                            /*columns=*/true,
                            rows,
                            cols),
               std::invalid_argument);
}

#else  // !defined(COLMAP_CUSIFT_ENABLED)

TEST(CusiftMatchingOptions, CheckFailsWithoutCusift) {
  EXPECT_FALSE(CreateCusiftMatchingOptions().Check());
}

TEST(CusiftFeatureMatcher, ReturnsNullWithoutCusift) {
  EXPECT_EQ(CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions()), nullptr);
}

#endif  // defined(COLMAP_CUSIFT_ENABLED)

}  // namespace
}  // namespace colmap
