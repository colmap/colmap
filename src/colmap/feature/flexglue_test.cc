// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/flexglue.h"

#include "colmap/feature/matcher.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/testing.h"

#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Synthetic scene of random colored rectangles on a smooth background. The two
// images are crops of the same canvas, offset by (kShiftX, kShiftY) pixels.
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kShiftX = 37;
constexpr int kShiftY = 21;

Bitmap CreateCanvas(int width, int height) {
  SetPRNGSeed(42);
  Bitmap canvas(width, height, /*as_rgb=*/true);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      canvas.SetPixel(
          x,
          y,
          BitmapColor<uint8_t>(static_cast<uint8_t>(x * 255 / width),
                               static_cast<uint8_t>(y * 255 / height),
                               128));
    }
  }
  for (int i = 0; i < 300; ++i) {
    const int size = RandomUniformInteger(8, 80);
    const int x0 = RandomUniformInteger(0, width - 1);
    const int y0 = RandomUniformInteger(0, height - 1);
    const BitmapColor<uint8_t> color(
        static_cast<uint8_t>(RandomUniformInteger(0, 255)),
        static_cast<uint8_t>(RandomUniformInteger(0, 255)),
        static_cast<uint8_t>(RandomUniformInteger(0, 255)));
    for (int y = y0; y < std::min(height, y0 + size); ++y) {
      for (int x = x0; x < std::min(width, x0 + size); ++x) {
        canvas.SetPixel(x, y, color);
      }
    }
  }
  return canvas;
}

Bitmap Crop(const Bitmap& canvas, int x0, int y0, int width, int height) {
  Bitmap crop(width, height, /*as_rgb=*/true);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      crop.SetPixel(x, y, *canvas.GetPixel(x0 + x, y0 + y));
    }
  }
  return crop;
}

// Matching all pairs through the full model is expensive on the CPU, so the
// model is shared across tests. It must be destroyed in TearDownTestSuite,
// since ONNX sessions must not be destroyed during static teardown.
class FlexGlueTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // CreateTestDir() requires a running test, so create the shared directory
    // manually, following the same naming convention.
    test_dir_ = std::make_unique<std::filesystem::path>(
        std::filesystem::temp_directory_path() / "colmap_test_data" /
        "FlexGlueTest");
    std::filesystem::remove_all(*test_dir_);
    std::filesystem::create_directories(*test_dir_);
    const Bitmap canvas =
        CreateCanvas(kWidth + kShiftX + 1, kHeight + kShiftY + 1);
    ASSERT_TRUE(
        Crop(canvas, 0, 0, kWidth, kHeight).Write(*test_dir_ / "image1.png"));
    ASSERT_TRUE(Crop(canvas, kShiftX, kShiftY, kWidth, kHeight)
                    .Write(*test_dir_ / "image2.png"));

    options_ =
        std::make_unique<FeatureMatchingOptions>(FeatureMatcherType::FLEXGLUE);
    options_->use_gpu = false;
    options_->flexglue->image_path = test_dir_->string();
    matcher_ = CreateFlexGlueFeatureMatcher(*options_);
  }

  static void TearDownTestSuite() {
    matcher_.reset();
    options_.reset();
    std::filesystem::remove_all(*test_dir_);
    test_dir_.reset();
  }

  // Keypoints on a regular grid in image1, and the same scene points in
  // image2, so that keypoint i of image1 corresponds to keypoint i of image2.
  static void CreateKeypoints(FeatureKeypoints* keypoints1,
                              FeatureKeypoints* keypoints2) {
    for (int y = 40; y < kHeight - 40; y += 16) {
      for (int x = 60; x < kWidth - 40; x += 16) {
        keypoints1->emplace_back(x + 0.5f, y + 0.5f);
        keypoints2->emplace_back(x - kShiftX + 0.5f, y - kShiftY + 0.5f);
      }
    }
  }

  static FeatureMatches Match(FeatureMatcher* matcher,
                              const FeatureKeypoints& keypoints1,
                              const FeatureKeypoints& keypoints2) {
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kPinhole, 500, kWidth, kHeight);
    const FeatureMatcher::Image image1{
        /*image_id=*/1,
        &camera,
        std::make_shared<FeatureKeypoints>(keypoints1),
        /*descriptors=*/nullptr,
        /*pose_prior=*/nullptr,
        /*name=*/"image1.png"};
    const FeatureMatcher::Image image2{
        /*image_id=*/2,
        &camera,
        std::make_shared<FeatureKeypoints>(keypoints2),
        /*descriptors=*/nullptr,
        /*pose_prior=*/nullptr,
        /*name=*/"image2.png"};
    FeatureMatches matches;
    matcher->Match(image1, image2, &matches);
    return matches;
  }

  static std::unique_ptr<std::filesystem::path> test_dir_;
  static std::unique_ptr<FeatureMatchingOptions> options_;
  static std::unique_ptr<FeatureMatcher> matcher_;
};

std::unique_ptr<std::filesystem::path> FlexGlueTest::test_dir_;
std::unique_ptr<FeatureMatchingOptions> FlexGlueTest::options_;
std::unique_ptr<FeatureMatcher> FlexGlueTest::matcher_;

TEST(FlexGlueMatchingOptions, Check) {
  FlexGlueMatchingOptions options;
  // The image path is required and must exist.
  EXPECT_FALSE(options.Check());
  options.image_path = (CreateTestDir() / "does_not_exist").string();
  EXPECT_FALSE(options.Check());
  options.image_path = CreateTestDir().string();
  EXPECT_TRUE(options.Check());
  options.min_covisibility = 1.5;
  EXPECT_FALSE(options.Check());
  options.min_covisibility = 0.8;
  EXPECT_TRUE(options.Check());
  options.early_stop_decay = 1.5;
  EXPECT_FALSE(options.Check());
  options.early_stop_decay = 0.5;
  EXPECT_TRUE(options.Check());
  options.filter_threshold = -0.1;
  EXPECT_FALSE(options.Check());
  options.filter_threshold = 0.1;
  options.max_distance_px = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(options.Check());
  // Non-positive values disable the distance check.
  options.max_distance_px = 0;
  EXPECT_TRUE(options.Check());
  options.max_distance_px = 3;
  options.feature_cache_size = 0;
  EXPECT_FALSE(options.Check());
  options.feature_cache_size = 128;
  options.descriptor_model_path.clear();
  EXPECT_FALSE(options.Check());
  options.descriptor_model_path = kDefaultFlexGlueDescriptorUri;
  options.matcher_model_path.clear();
  EXPECT_FALSE(options.Check());
  options.matcher_model_path = kDefaultFlexGlueMatcherUri;
  // Without the float16 matcher, the float32 matcher is used.
  options.matcher_model_path_fp16.clear();
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.num_workers_per_gpu, 4);
  options.num_workers_per_gpu = 0;
  EXPECT_FALSE(options.Check());
  options.num_workers_per_gpu = 4;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.num_inferences_per_gpu, 2);
  options.num_inferences_per_gpu = 0;
  EXPECT_FALSE(options.Check());
  options.num_inferences_per_gpu = 1;
  EXPECT_TRUE(options.Check());
}

TEST(FlexGlueMatchingOptions, FeatureMatchingOptions) {
  FeatureMatchingOptions options(FeatureMatcherType::FLEXGLUE);
  options.flexglue->image_path = CreateTestDir().string();
  EXPECT_TRUE(options.Check());
  // FlexGlue only uses the keypoints, and the matching workers then neither
  // require nor load descriptors.
  EXPECT_FALSE(options.UsesDescriptors());
  EXPECT_TRUE(FeatureMatchingOptions(FeatureMatcherType::SIFT_BRUTEFORCE)
                  .UsesDescriptors());
  // Guided matching is not supported, which is rejected before loading the
  // models.
  options.guided_matching = true;
  EXPECT_FALSE(options.Check());
}

TEST_F(FlexGlueTest, MatchTranslatedImages) {
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  const FeatureMatches matches = Match(matcher_.get(), keypoints1, keypoints2);

  int num_correct = 0;
  for (const FeatureMatch& match : matches) {
    EXPECT_LT(match.point2D_idx1, keypoints1.size());
    EXPECT_LT(match.point2D_idx2, keypoints2.size());
    if (match.point2D_idx1 == match.point2D_idx2) {
      ++num_correct;
    }
  }
  EXPECT_GT(matches.size(), keypoints1.size() / 2);
  EXPECT_GT(num_correct, 0.95 * matches.size());

  // Matching the same pair again reuses the cached dense features and must
  // give identical results.
  EXPECT_EQ(Match(matcher_.get(), keypoints1, keypoints2), matches);

  // Mutual nearest-neighbor distances and max_distance_px operate in the
  // resized descriptor input frame, so anisotropically rescaling an image's
  // camera dimensions and keypoints must leave the matches unchanged.
  FeatureKeypoints rescaled_keypoints2;
  rescaled_keypoints2.reserve(keypoints2.size());
  for (const FeatureKeypoint& kp : keypoints2) {
    rescaled_keypoints2.emplace_back(4.f * kp.x, 2.f * kp.y);
  }
  const Camera camera1 = Camera::CreateFromModelId(
      1, CameraModelId::kPinhole, 500, kWidth, kHeight);
  const Camera camera2 = Camera::CreateFromModelId(
      2, CameraModelId::kPinhole, 1500, 4 * kWidth, 2 * kHeight);
  const FeatureMatcher::Image image1{
      /*image_id=*/1,
      &camera1,
      std::make_shared<FeatureKeypoints>(keypoints1),
      /*descriptors=*/nullptr,
      /*pose_prior=*/nullptr,
      /*name=*/"image1.png"};
  const FeatureMatcher::Image image2{
      /*image_id=*/2,
      &camera2,
      std::make_shared<FeatureKeypoints>(rescaled_keypoints2),
      /*descriptors=*/nullptr,
      /*pose_prior=*/nullptr,
      /*name=*/"image2.png"};
  FeatureMatches rescaled_matches;
  matcher_->Match(image1, image2, &rescaled_matches);
  EXPECT_EQ(rescaled_matches, matches);
}

TEST_F(FlexGlueTest, HeavyMinCovisibilityRejectsAllMatches) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (std::getenv("COLMAP_TEST_HEAVY") == nullptr) {
    GTEST_SKIP() << "Set COLMAP_TEST_HEAVY=1 to run tests that load another "
                    "FlexGlue matcher.";
  }
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  FeatureMatchingOptions options = *options_;
  // The covisibility score is a probability in [0, 1), so no pair passes.
  // Disable early stopping, so that only the final pair rejection applies.
  options.flexglue->min_covisibility = 1.0;
  options.flexglue->early_stop_decay = 1.0;
  std::unique_ptr<FeatureMatcher> matcher =
      CreateFlexGlueFeatureMatcher(options);
  EXPECT_TRUE(Match(matcher.get(), keypoints1, keypoints2).empty());
}

TEST_F(FlexGlueTest, HeavyEarlyStopping) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (std::getenv("COLMAP_TEST_HEAVY") == nullptr) {
    GTEST_SKIP() << "Set COLMAP_TEST_HEAVY=1 to run tests that load another "
                    "FlexGlue matcher.";
  }
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  // Early stopping (default early_stop_decay = 0.5) must not reject this
  // clearly covisible pair, i.e., give the same matches as without it. That it
  // stops on non-covisible pairs exactly as FlexGlue does is checked by
  // onnx/check.py in the FlexGlue repository.
  FeatureMatchingOptions options = *options_;
  options.flexglue->early_stop_decay = 1.0;
  std::unique_ptr<FeatureMatcher> matcher =
      CreateFlexGlueFeatureMatcher(options);
  const FeatureMatches matches = Match(matcher.get(), keypoints1, keypoints2);
  EXPECT_FALSE(matches.empty());
  EXPECT_EQ(Match(matcher_.get(), keypoints1, keypoints2), matches);
}

TEST_F(FlexGlueTest, EmptyKeypoints) {
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  EXPECT_TRUE(Match(matcher_.get(), {}, keypoints2).empty());
  EXPECT_TRUE(Match(matcher_.get(), keypoints1, {}).empty());
}

TEST_F(FlexGlueTest, MissingImageThrows) {
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kPinhole, 500, kWidth, kHeight);
  const FeatureMatcher::Image image1{
      /*image_id=*/1,
      &camera,
      std::make_shared<FeatureKeypoints>(keypoints1),
      /*descriptors=*/nullptr,
      /*pose_prior=*/nullptr,
      /*name=*/"does_not_exist.png"};
  const FeatureMatcher::Image image2{
      /*image_id=*/2,
      &camera,
      std::make_shared<FeatureKeypoints>(keypoints2),
      /*descriptors=*/nullptr,
      /*pose_prior=*/nullptr,
      /*name=*/"image2.png"};
  FeatureMatches matches;
  try {
    matcher_->Match(image1, image2, &matches);
    ADD_FAILURE() << "Expected an exception";
  } catch (const std::exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("Failed to read image"));
    EXPECT_THAT(e.what(), testing::HasSubstr("does_not_exist.png"));
  }
}

TEST_F(FlexGlueTest, ConcurrentMatchThreadSafety) {
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  CreateKeypoints(&keypoints1, &keypoints2);
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kPinhole, 500, kWidth, kHeight);
  const auto kpts1_ptr = std::make_shared<FeatureKeypoints>(keypoints1);
  const auto kpts2_ptr = std::make_shared<FeatureKeypoints>(keypoints2);

  const FeatureMatches expected_matches =
      Match(matcher_.get(), keypoints1, keypoints2);
  ASSERT_FALSE(expected_matches.empty());
  FeatureMatches expected_swapped_matches;
  matcher_->Match(FeatureMatcher::Image{/*image_id=*/2,
                                        &camera,
                                        kpts2_ptr,
                                        /*descriptors=*/nullptr,
                                        /*pose_prior=*/nullptr,
                                        /*name=*/"image2.png"},
                  FeatureMatcher::Image{/*image_id=*/1,
                                        &camera,
                                        kpts1_ptr,
                                        /*descriptors=*/nullptr,
                                        /*pose_prior=*/nullptr,
                                        /*name=*/"image1.png"},
                  &expected_swapped_matches);
  ASSERT_FALSE(expected_swapped_matches.empty());

  // Fresh image names ensure all threads hit cold entries in
  // ThreadSafeLRUCache simultaneously in the first round; after a barrier,
  // the second round re-matches the warm entries (in both pair orders).
  std::filesystem::copy_file(*test_dir_ / "image1.png",
                             *test_dir_ / "concurrent_image1.png",
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(*test_dir_ / "image2.png",
                             *test_dir_ / "concurrent_image2.png",
                             std::filesystem::copy_options::overwrite_existing);

  constexpr int kNumThreads = 3;
  int barrier_arrived = 0;
  int barrier_phase = 0;
  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  auto arrive_and_wait = [&]() {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    const int phase = barrier_phase;
    if (++barrier_arrived == kNumThreads) {
      barrier_arrived = 0;
      ++barrier_phase;
      lock.unlock();
      barrier_cv.notify_all();
    } else {
      barrier_cv.wait(lock, [&]() { return barrier_phase != phase; });
    }
  };

  std::vector<FeatureMatches> cold_results(kNumThreads);
  std::vector<FeatureMatches> warm_results(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      const FeatureMatcher::Image image1{/*image_id=*/1,
                                         &camera,
                                         kpts1_ptr,
                                         /*descriptors=*/nullptr,
                                         /*pose_prior=*/nullptr,
                                         /*name=*/"concurrent_image1.png"};
      const FeatureMatcher::Image image2{/*image_id=*/2,
                                         &camera,
                                         kpts2_ptr,
                                         /*descriptors=*/nullptr,
                                         /*pose_prior=*/nullptr,
                                         /*name=*/"concurrent_image2.png"};
      arrive_and_wait();
      matcher_->Match(image1, image2, &cold_results[t]);
      arrive_and_wait();
      if (t % 2 == 0) {
        matcher_->Match(image1, image2, &warm_results[t]);
      } else {
        matcher_->Match(image2, image1, &warm_results[t]);
      }
    });
  }

  for (std::thread& thread : threads) {
    thread.join();
  }
  for (int t = 0; t < kNumThreads; ++t) {
    EXPECT_EQ(cold_results[t], expected_matches);
    EXPECT_EQ(warm_results[t],
              t % 2 == 0 ? expected_matches : expected_swapped_matches);
  }
}

TEST_F(FlexGlueTest, GuidedMatchingNotSupported) {
  TwoViewGeometry two_view_geometry;
  EXPECT_ANY_THROW(matcher_->MatchGuided(4.0, {}, {}, &two_view_geometry));
}

}  // namespace
}  // namespace colmap
