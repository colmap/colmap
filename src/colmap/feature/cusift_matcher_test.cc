// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

FeatureMatchingOptions CreateCusiftMatchingOptions() {
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  return options;
}

TEST(CusiftMatchingOptions, DefaultsMatchTheExactReference) {
  // Bidirectional ratio test on angles, as FindBestMatchesOneWayBruteForce.
  const CusiftMatchingOptions options;
  EXPECT_TRUE(options.cross_check_reverse_ratio);
  EXPECT_EQ(options.distance_type, CusiftDistanceType::ANGULAR);
}

// Options pass Check() only in a build with cusift; without it, Check() rejects
// the cusift types before reaching cusift's own checks.
#if defined(COLMAP_CUSIFT_ENABLED)
TEST(CusiftMatchingOptions, ChecksSiftThresholds) {
  // The ratio and distance thresholds come from the SIFT options, and are
  // checked for CUSIFT_GEMM too.
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  EXPECT_TRUE(options.Check());
  options.sift->max_ratio = -1;
  EXPECT_FALSE(options.Check());
  options = CreateCusiftMatchingOptions();
  options.sift->max_distance = -1;
  EXPECT_FALSE(options.Check());
}

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

// Helper to create empty descriptors for testing. Mirrors
// sift_test.cc's CreateEmptyDescriptors -- duplicated rather than shared,
// same as cusift_test.cc does not share helpers with extractor_test.cc.
FeatureDescriptors CreateEmptyDescriptors() {
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsData(0, 128));
}

// Helper to create reversed descriptors for testing matcher symmetry. Mirrors
// sift_test.cc's CreateReversedDescriptors.
FeatureDescriptors CreateReversedDescriptors(const FeatureDescriptors& src) {
  return FeatureDescriptors(src.type, src.data.colwise().reverse());
}

// Mirrors sift_test.cc's CreateRandomFeatureDescriptors: each row is a sparse
// random unit vector (10 of 128 dims set to 1, then L2-normalized and
// quantized), so distinct rows are near-orthogonal and cross-row matches are
// clearly separated from a row's match with itself.
FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloatData descriptors_float =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptors(
      FeatureExtractorType::SIFT,
      FeatureDescriptorsToUnsignedByte(descriptors_float));
}

// Mirrors sift_test.cc's ExpectReversedMatches.
void ExpectReversedMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);
}

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
    options.sift->cross_check = false;
    options.sift->max_ratio = 1.0;  // a single candidate has no runner-up
    for (const double factor : {1.05, 0.95}) {
      options.sift->max_distance = factor * distance;
      auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
      FeatureMatches matches;
      matcher->Match(image1, image2, &matches);
      EXPECT_EQ(matches.size(), factor > 1 ? 1 : 0)
          << "distance_type=" << distance_type << " distance=" << distance
          << " max_distance=" << options.sift->max_distance;
    }
  }
}

TEST(CusiftFeatureMatcher, MaxNumMatchesTruncatesByIndexOrder) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  constexpr size_t kNumFeatures = 20;
  constexpr int kMaxNumMatches = 5;

  FeatureDescriptors desc = CreateRandomFeatureDescriptors(kNumFeatures);
  const FeatureMatcher::Image image1 = MakeImage(1, desc);
  const FeatureMatcher::Image image2 = MakeImage(2, desc);

  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.sift->cross_check = false;
  options.sift->max_ratio = 1.0;  // every row self-matches trivially
  options.max_num_matches = kMaxNumMatches;

  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Every one of the kNumFeatures rows qualifies (self-match, distance 0),
  // so the cap binds: survivors must be exactly the first kMaxNumMatches by
  // ascending point2D_idx1 -- SiftMatchGPU's own cap-binds rule (see
  // cusift_matcher.h) -- NOT a distance-sorted top-K (all distances are tied
  // at 0 here, so a top-K would be indistinguishable from this by distance
  // alone; the index-order assertion is what actually exercises the rule).
  ASSERT_EQ(matches.size(), static_cast<size_t>(kMaxNumMatches));
  for (int i = 0; i < kMaxNumMatches; ++i) {
    EXPECT_EQ(matches[i].point2D_idx1, static_cast<point2D_t>(i));
    EXPECT_EQ(matches[i].point2D_idx2, static_cast<point2D_t>(i));
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

// Differential fuzz against COLMAP's EXACT reference matcher, on descriptors
// with NON-UNIFORM norms, where ranking by raw inner product and ranking by
// normalized distance disagree.
//
// The reference is deliberately `cpu_brute_force_matcher = true`. The default
// CPU path builds a faiss IVF index and probes 8 of ~400 cells, i.e. it is
// APPROXIMATE despite `SIFT_BRUTEFORCE` in the enum name, and it also applies a
// different distance convention, so it is not a bit-exact reference.
//
// Bit-exactness is the ideal property to fuzz: the assertion is equality, with
// no tolerance to tune.
TEST(CusiftMatcherParity, DifferentialAgainstExactCpuBruteForce) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";

  size_t total_reference_matches = 0;
  for (int seed = 1; seed <= 12; ++seed) {
    for (const size_t n : {2u, 3u, 17u, 64u, 200u}) {
      const FeatureDescriptors da = CreateDescriptorsWithVaryingNorms(n, seed);
      // db must contain genuine correspondences, or both matchers return the
      // empty set and the equality assertion passes vacuously. Independent
      // random sparse descriptors are near-orthogonal and produce almost no
      // matches. Perturbed copies of da give real matches AND keep the near-tie
      // population that makes the test sensitive.
      FeatureDescriptorsFloatData f = da.ToFloat().data;
      for (Eigen::Index r = 0; r < f.rows(); ++r) {
        const int dim = static_cast<int>((r * 37 + seed) % 128);
        f(r, dim) += 6.0f + static_cast<float>((r + seed) % 5);
      }
      const FeatureDescriptors db(FeatureExtractorType::SIFT,
                                  f.cast<uint8_t>());
      const auto img1 = MakeImage(1, da);
      const auto img2 = MakeImage(2, db);

      FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
      auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));

      FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
      ref.use_gpu = false;
      ref.sift->cpu_brute_force_matcher =
          true;  // EXACT, not the faiss IVF path
      ref.sift->max_ratio = cu.sift->max_ratio;
      ref.sift->max_distance = cu.sift->max_distance;
      ref.sift->cross_check = cu.sift->cross_check;
      auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

      const auto got = MatchToSet(*cu_matcher, img1, img2);
      const auto want = MatchToSet(*ref_matcher, img1, img2);
      total_reference_matches += want.size();
      EXPECT_EQ(got, want) << "cusift diverged from the exact reference at n="
                           << n << " seed=" << seed << " (got " << got.size()
                           << " vs " << want.size() << " matches)";
    }
  }
  // Non-vacuity guard: if the fixtures stop producing matches this test
  // silently becomes "empty == empty" and stops testing anything at all.
  EXPECT_GT(total_reference_matches, 100u)
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

  FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
  auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));
  FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
  ref.use_gpu = false;
  ref.sift->cpu_brute_force_matcher = true;
  auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

  const auto got = MatchToSet(*cu_matcher, img1, img2);
  const auto want = MatchToSet(*ref_matcher, img1, img2);
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

  FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
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
