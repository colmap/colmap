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

#include "colmap/feature/cusift_matcher.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"

#include <algorithm>
#include <numeric>
#include <set>

#include <gtest/gtest.h>

namespace colmap {
namespace {

FeatureMatchingOptions CreateCusiftMatchingOptions() {
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  return options;
}

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

#if defined(COLMAP_CUDA_ENABLED)

TEST(CusiftMatchingOptions, Check) {
  CusiftMatchingOptions options;
  EXPECT_TRUE(options.Check());
  options.max_ratio = -1;
  EXPECT_FALSE(options.Check());
  options.max_ratio = 0.8;
  options.max_distance = -1;
  EXPECT_FALSE(options.Check());
}

TEST(CusiftMatchingOptions, ReverseRatioDefaultsOnForParity) {
  // Default must be the SiftGPU/COLMAP-CPU-equivalent bidirectional test:
  // the weaker mutual-best-only mode is opt-in, never the default.
  const CusiftMatchingOptions options;
  EXPECT_TRUE(options.cross_check_reverse_ratio);
}

// Both cross-check modes run the same assertions where behaviour must agree;
// where they differ is asserted explicitly in ReverseRatioIsStricter below.
class CusiftCrossCheckModeTest : public ::testing::TestWithParam<bool> {};

TEST_P(CusiftCrossCheckModeTest, SelfMatchIdentity) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check_reverse_ratio = GetParam();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  FeatureDescriptors desc = CreateRandomFeatureDescriptors(16);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  // Each descriptor's unique nearest neighbour is itself, in either mode.
  EXPECT_GT(matches.size(), 0);
  for (const auto& m : matches) {
    EXPECT_EQ(m.point2D_idx1, m.point2D_idx2);
  }
}

TEST_P(CusiftCrossCheckModeTest, EmptyAndReversedInputs) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check_reverse_ratio = GetParam();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  FeatureMatches matches;
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

TEST(CusiftFeatureMatcher, RequiresGpu) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  EXPECT_TRUE(options.Check());
  // cusift has no CPU implementation, so this must be rejected as a bad
  // option rather than surfacing later as an unknown-matcher-type error.
  options.use_gpu = false;
  EXPECT_FALSE(options.Check());
}

TEST(CusiftFeatureMatcher, RejectsGuidedMatching) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.guided_matching = true;
  EXPECT_THROW(CreateCusiftFeatureMatcher(options), std::exception);
}

TEST(CusiftFeatureMatcher, Nominal) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  FeatureMatches matches;

  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  // Same query descriptors again -- exercises the thread_local
  // GemmMatcherContext's grow-only buffer reuse path, not just first-call
  // allocation.
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

TEST(CusiftFeatureMatcher, TypeMismatch) {
  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureMatches matches;
  EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
               std::exception);
}

// Retained alongside ReverseRatioRejectsAmbiguousReverseDirection, which is
// strictly stronger. This one asserts only the subset invariant and passes
// vacuously when the two modes agree, so it is not sufficient on its own --
// but tests are additive: a weak invariant is still an invariant.
TEST(CusiftFeatureMatcher, ReverseRatioIsStricter) {
  // The defining behavioural difference: the bidirectional ratio test can only
  // ever remove matches that mutual-best-only would keep, never add any. On
  // synthetic near-orthogonal descriptors the two usually agree, so this
  // asserts the invariant (subset) rather than a strict inequality, which only
  // shows up on real repetitive structure (see
  // src/thirdparty/cusift/README.md).
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  FeatureDescriptors d1 = CreateRandomFeatureDescriptors(64);
  FeatureDescriptors d2 = CreateReversedDescriptors(d1);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(d1)};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(d2)};

  auto run = [&](bool reverse_ratio) {
    FeatureMatchingOptions options = CreateCusiftMatchingOptions();
    options.cusift->cross_check_reverse_ratio = reverse_ratio;
    auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
    FeatureMatches matches;
    matcher->Match(image1, image2, &matches);
    std::set<std::pair<point2D_t, point2D_t>> out;
    for (const auto& m : matches) {
      out.emplace(m.point2D_idx1, m.point2D_idx2);
    }
    return out;
  };

  const auto strict = run(/*reverse_ratio=*/true);
  const auto loose = run(/*reverse_ratio=*/false);
  EXPECT_LE(strict.size(), loose.size());
  for (const auto& m : strict) {
    EXPECT_TRUE(loose.count(m) > 0)
        << "bidirectional test kept a match mutual-best-only rejected, which "
           "should be impossible";
  }
}

TEST(CusiftFeatureMatcher, MaxDistanceFiltersCorrectly) {
  // Two descriptors with disjoint support sets (no shared dims) are far
  // apart in unit-vector L2 space (distance close to sqrt(2) ~= 1.414 for
  // orthogonal unit vectors); two copies of the *same* descriptor are at
  // distance 0. Use that gap rather than a hand-derived exact value, since
  // uint8 quantization perturbs the exact distance slightly (see
  // cusift_matcher.h's quantization caveat).
  FeatureDescriptors desc = CreateRandomFeatureDescriptors(1);
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image_self = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};
  const FeatureMatcher::Image image_copy = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};

  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check = false;
  options.cusift->max_ratio = 1.0;  // effectively disable the ratio test

  FeatureMatches matches;

  // Distance ~0 must survive a small max_distance.
  options.cusift->max_distance = 0.01;
  auto matcher_tight = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
  matcher_tight->Match(image_self, image_copy, &matches);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 0);
}

TEST(CusiftFeatureMatcher, MaxNumMatchesTruncatesByIndexOrder) {
  constexpr size_t kNumFeatures = 20;
  constexpr int kMaxNumMatches = 5;

  FeatureDescriptors desc = CreateRandomFeatureDescriptors(kNumFeatures);
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc)};

  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check = false;
  options.cusift->max_ratio = 1.0;  // every row self-matches trivially
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

FeatureMatcher::Image MakeImage(image_t id,
                                const Camera& camera,
                                const FeatureDescriptors& desc) {
  return FeatureMatcher::Image{id,
                               &camera,
                               /*keypoints=*/nullptr,
                               std::make_shared<FeatureDescriptors>(desc)};
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
  // THE case the old subset-only assertion could not see. Two near-duplicate
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

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const auto img1 = MakeImage(1, camera, d1);
  const auto img2 = MakeImage(2, camera, d2);

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
// The fixture must also DISCRIMINATE the old rule from the new one. ColTop2
// block-strides rows over 256 threads, so the old code resolved ties to the
// lowest threadIdx.x == lowest (r % 256), not the lowest r. Rows 0 and 1 would
// map to threads 0 and 1 and both rules would agree -- a useless fixture. Rows
// 45 and 300 map to threads 45 and 44, so the old rule picks row 300 while the
// canonical rule picks row 45.
//
// reverse_ratio is disabled here on purpose: two tied candidates make the
// column's own best == second, which its ratio test would reject, removing the
// very match this test needs to observe.
TEST(CusiftFeatureMatcher, ColumnTieBreakPrefersLowerRowIndex) {
  constexpr int kN1 = 320;       // > 300 so row 300 exists
  constexpr int kTieLow = 45;    // -> thread 45
  constexpr int kTieHigh = 300;  // -> thread 44, so the OLD rule would win here
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

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const auto img1 = MakeImage(1, camera, d1);
  const auto img2 = MakeImage(2, camera, d2);

  FeatureMatchingOptions options = CreateCusiftMatchingOptions();
  options.cusift->cross_check = true;
  options.cusift->cross_check_reverse_ratio = false;
  auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));

  FeatureMatches matches;
  matcher->Match(img1, img2, &matches);

  ASSERT_EQ(matches.size(), 1)
      << "expected exactly one surviving cross-checked "
         "match for the single image-2 descriptor";
  EXPECT_EQ(matches[0].point2D_idx1, static_cast<point2D_t>(kTieLow))
      << "column tie must resolve to the LOWER row index (" << kTieLow
      << "), not the one that happens to land on a lower thread (" << kTieHigh
      << ")";
  EXPECT_EQ(matches[0].point2D_idx2, 0);
}

// Same tie, opposite key space: the key changes the compared VALUE but must not
// change the tie rule.
TEST(CusiftFeatureMatcher, ColumnTieBreakIsKeyModeIndependent) {
  constexpr int kN1 = 320;
  constexpr int kTieLow = 45;
  constexpr int kTieHigh = 300;

  FeatureDescriptors d1 = CreateRandomFeatureDescriptors(kN1);
  d1.data.row(kTieHigh) = d1.data.row(kTieLow);
  FeatureDescriptorsData d2data(1, 128);
  d2data.row(0) = d1.data.row(kTieLow);
  const FeatureDescriptors d2(FeatureExtractorType::SIFT, d2data);

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const auto img1 = MakeImage(1, camera, d1);
  const auto img2 = MakeImage(2, camera, d2);

  for (const auto dt : {CusiftDistanceType::ANGULAR, CusiftDistanceType::L2}) {
    FeatureMatchingOptions options = CreateCusiftMatchingOptions();
    options.cusift->distance_type = dt;
    options.cusift->cross_check = true;
    options.cusift->cross_check_reverse_ratio = false;
    auto matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options));
    FeatureMatches matches;
    matcher->Match(img1, img2, &matches);
    ASSERT_EQ(matches.size(), 1) << "distance_type=" << static_cast<int>(dt);
    EXPECT_EQ(matches[0].point2D_idx1, static_cast<point2D_t>(kTieLow))
        << "tie rule must not depend on the key space; distance_type="
        << static_cast<int>(dt);
  }
}

// Differential fuzz against COLMAP's EXACT reference matcher. This is the test
// that would have caught the normalization bug: it uses descriptors with
// NON-UNIFORM norms, where ranking by raw inner product and ranking by
// normalized distance disagree.
//
// The reference is deliberately `cpu_brute_force_matcher = true`. The default
// CPU path builds a faiss IVF index and probes 8 of ~400 cells, i.e. it is
// APPROXIMATE despite `SIFT_BRUTEFORCE` in the enum name, and it also applies a
// different distance convention -- comparing against it silently validates the
// wrong thing (measured: 0.9757 agreement with the exact matcher).
//
// Bit-exactness is the ideal property to fuzz: the assertion is equality, with
// no tolerance to tune.
TEST(CusiftMatcherParity, DifferentialAgainstExactCpuBruteForce) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  size_t total_reference_matches = 0;
  for (int seed = 1; seed <= 12; ++seed) {
    for (const size_t n : {2u, 3u, 17u, 64u, 200u}) {
      const FeatureDescriptors da = CreateDescriptorsWithVaryingNorms(n, seed);
      // db must contain genuine correspondences, or both matchers return the
      // empty set and the equality assertion passes vacuously. Independent
      // random sparse descriptors are near-orthogonal and produce almost no
      // matches -- which is exactly how an earlier version of this test managed
      // to survive a mutation that broke the ranking key. Perturbed copies of
      // da give real matches AND keep the near-tie population that makes the
      // test sensitive.
      FeatureDescriptorsFloatData f = da.ToFloat().data;
      for (Eigen::Index r = 0; r < f.rows(); ++r) {
        const int dim = static_cast<int>((r * 37 + seed) % 128);
        f(r, dim) += 6.0f + static_cast<float>((r + seed) % 5);
      }
      const FeatureDescriptors db(FeatureExtractorType::SIFT,
                                  f.cast<uint8_t>());
      const auto img1 = MakeImage(1, camera, da);
      const auto img2 = MakeImage(2, camera, db);

      FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
      auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));

      FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
      ref.use_gpu = false;
      ref.sift->cpu_brute_force_matcher =
          true;  // EXACT, not the faiss IVF path
      ref.sift->max_ratio = cu.cusift->max_ratio;
      ref.sift->max_distance = cu.cusift->max_distance;
      ref.sift->cross_check = cu.cusift->cross_check;
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
// invisible to positive-pair tests by construction: the missing reverse-ratio
// test inflated spurious matches 2.8x on non-overlapping real imagery while
// every unit test still passed.
TEST(CusiftMatcherParity, NegativeControlUnrelatedDescriptors) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  // Independently generated sets: nothing should legitimately correspond.
  const FeatureDescriptors da = CreateDescriptorsWithVaryingNorms(150, 21);
  const FeatureDescriptors db = CreateDescriptorsWithVaryingNorms(150, 99);
  const auto img1 = MakeImage(1, camera, da);
  const auto img2 = MakeImage(2, camera, db);

  FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
  auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));
  FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
  ref.use_gpu = false;
  ref.sift->cpu_brute_force_matcher = true;
  auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

  const auto got = MatchToSet(*cu_matcher, img1, img2);
  const auto want = MatchToSet(*ref_matcher, img1, img2);
  // Exact agreement is the spec. The looser "within 1.2x of the reference
  // count" form is what to fall back on if a future backend is only
  // approximately equivalent -- it still catches a 2.8x inflation.
  EXPECT_EQ(got, want);
  EXPECT_LE(got.size(), static_cast<size_t>(1.2 * want.size()) + 1)
      << "spurious matches on unrelated content inflated beyond the reference";
}

// Bug 2, surgically. Descriptors whose norms deliberately miss COLMAP's nominal
// 512 (here ~400 and ~600), placed so that ranking by RAW inner product and
// ranking by NORMALIZED distance choose different winners. Under per-row
// normalization the decision flips; under the reference's fixed-512 convention
// it does not. Same boundary-test discipline as MaxDistanceFiltersCorrectly,
// applied to the ratio/cross-check semantics rather than only to max_distance.
TEST(CusiftMatcherParity, NonCanonicalNormsDoNotChangeTheDecision) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

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
  const auto img1 = MakeImage(1, camera, dq);
  const auto img2 = MakeImage(2, camera, dc);

  FeatureMatchingOptions cu = CreateCusiftMatchingOptions();
  cu.cusift->max_ratio = 0.99;  // keep the ratio test from masking the choice
  auto cu_matcher = THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cu));
  FeatureMatchingOptions ref(FeatureMatcherType::SIFT_BRUTEFORCE);
  ref.use_gpu = false;
  ref.sift->cpu_brute_force_matcher = true;
  ref.sift->max_ratio = 0.99;
  auto ref_matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(ref));

  EXPECT_EQ(MatchToSet(*cu_matcher, img1, img2),
            MatchToSet(*ref_matcher, img1, img2))
      << "the winning candidate changed under non-canonical descriptor norms";
}

TEST(CusiftMatcherParity, MatchAgreementSmokeCheck) {
  // Lightweight smoke check on a small synthetic set. Match-set agreement on
  // real images is measured outside the unit tests (see
  // src/thirdparty/cusift/README.md): per-pair Jaccard is noisy on small match
  // counts, so this only checks that the two matchers agree on the trivial
  // reversed-descriptor case, not a statistically meaningful Jaccard bound.
  FeatureDescriptors desc1 = CreateRandomFeatureDescriptors(10);
  FeatureDescriptors desc2 = CreateReversedDescriptors(desc1);
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc1)};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(desc2)};

  FeatureMatchingOptions cusift_options = CreateCusiftMatchingOptions();
  auto cusift_matcher =
      THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(cusift_options));
  FeatureMatches cusift_matches;
  cusift_matcher->Match(image1, image2, &cusift_matches);

  FeatureMatchingOptions sift_options(FeatureMatcherType::SIFT_BRUTEFORCE);
  sift_options.use_gpu = false;
  sift_options.sift->cpu_brute_force_matcher = true;
  auto sift_matcher =
      THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(sift_options));
  FeatureMatches sift_matches;
  sift_matcher->Match(image1, image2, &sift_matches);

  auto ToPairSet = [](const FeatureMatches& matches) {
    std::set<std::pair<point2D_t, point2D_t>> pairs;
    for (const auto& m : matches) {
      pairs.emplace(m.point2D_idx1, m.point2D_idx2);
    }
    return pairs;
  };
  const auto cusift_pairs = ToPairSet(cusift_matches);
  const auto sift_pairs = ToPairSet(sift_matches);

  std::vector<std::pair<point2D_t, point2D_t>> intersection;
  std::set_intersection(cusift_pairs.begin(),
                        cusift_pairs.end(),
                        sift_pairs.begin(),
                        sift_pairs.end(),
                        std::back_inserter(intersection));
  std::vector<std::pair<point2D_t, point2D_t>> union_set;
  std::set_union(cusift_pairs.begin(),
                 cusift_pairs.end(),
                 sift_pairs.begin(),
                 sift_pairs.end(),
                 std::back_inserter(union_set));
  ASSERT_FALSE(union_set.empty());
  const double jaccard =
      static_cast<double>(intersection.size()) / union_set.size();
  EXPECT_GE(jaccard, 0.995);
}

#else  // !defined(COLMAP_CUDA_ENABLED)

TEST(CusiftFeatureMatcher, ReturnsNullWithoutCuda) {
  EXPECT_EQ(CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions()), nullptr);
}

#endif  // defined(COLMAP_CUDA_ENABLED)

}  // namespace
}  // namespace colmap
