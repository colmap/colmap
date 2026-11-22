// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/controllers/matcher_cache.h"
#include "colmap/feature/aliked.h"
#include "colmap/feature/flexglue.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/testing.h"

#include <functional>
#include <random>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  std::filesystem::path test_dir;
  std::shared_ptr<Database> database;
  std::shared_ptr<FeatureMatcherCache> cache;
  std::vector<image_t> image_ids;
};

TestData CreateTestData(int num_images) {
  TestData data;
  data.test_dir = CreateTestDir();
  const auto database_path = data.test_dir / "database.db";
  data.database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = num_images;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 20;
  options.num_points2D_without_point3D = 3;
  SynthesizeDataset(options, &reconstruction, data.database.get());

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  data.image_ids = data.cache->GetImageIds();
  return data;
}

// Message of the exception thrown by the function, or empty if none.
std::string ThrownMessage(const std::function<void()>& func) {
  try {
    func();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

FeatureMatchingOptions DefaultMatchingOptions() {
  FeatureMatchingOptions options;
  options.use_gpu = false;
  options.num_threads = 1;
  return options;
}

std::vector<std::pair<image_t, image_t>> AllPairs(
    const std::vector<image_t>& image_ids) {
  std::vector<std::pair<image_t, image_t>> pairs;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      pairs.emplace_back(image_ids[i], image_ids[j]);
    }
  }
  return pairs;
}

// A pair with 40 exact translation inliers among 100 matches (true ratio
// 0.4): with min_inlier_ratio=0.5 the ratio recheck -- not RANSAC failure --
// rejects it, so the estimator returns DEGENERATE *with* inliers.
struct LowRatioPairData {
  TestData data;
  image_t image_id1;
  image_t image_id2;
};

LowRatioPairData CreateLowRatioPairData() {
  LowRatioPairData pair_data;
  TestData& data = pair_data.data;
  data.test_dir = CreateTestDir();
  data.database = Database::Open(data.test_dir / "database.db");

  // Two distinct uncalibrated pinhole cameras -> uncalibrated path.
  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1000, 1000, 1000);
  camera1.camera_id = data.database->WriteCamera(camera1);
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1000, 1000, 1000);
  camera2.camera_id = data.database->WriteCamera(camera2);

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera1.camera_id);
  const image_t image_id1 = data.database->WriteImage(image1);
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera2.camera_id);
  const image_t image_id2 = data.database->WriteImage(image2);

  std::mt19937 prng(42);
  std::uniform_real_distribution<float> uniform(100.f, 900.f);
  constexpr int kNumInliers = 40;
  constexpr int kNumMatches = 100;
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  FeatureMatches matches;
  for (int i = 0; i < kNumMatches; ++i) {
    const float x1 = uniform(prng);
    const float y1 = uniform(prng);
    float x2 = x1 + 20.f;
    float y2 = y1 + 10.f;
    if (i >= kNumInliers) {
      x2 = uniform(prng);
      y2 = uniform(prng);
    }
    keypoints1.emplace_back(x1, y1);
    keypoints2.emplace_back(x2, y2);
    matches.emplace_back(i, i);
  }
  data.database->WriteKeypoints(image_id1, keypoints1);
  data.database->WriteKeypoints(image_id2, keypoints2);
  data.database->WriteMatches(image_id1, image_id2, matches);

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  pair_data.image_id1 = image_id1;
  pair_data.image_id2 = image_id2;
  return pair_data;
}

TwoViewGeometryOptions LowRatioGeometryOptions() {
  TwoViewGeometryOptions options;
  options.min_inlier_ratio = 0.5;
  options.detect_watermark = false;
  options.ransac_options.random_seed = 42;
  options.ransac_options.min_num_trials = 1000;
  return options;
}

// Match pairs without geometric verification, then clear TVGs.
// Leaves matches in the database ready for a GeometricVerifierController.
void MatchPairsWithoutVerification(
    TestData& data, const std::vector<std::pair<image_t, image_t>>& pairs) {
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController matcher(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(matcher.Setup());
  matcher.Match(pairs);
  data.database->ClearTwoViewGeometries();
}

TEST(FeatureMatcherController, MatchEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Matching empty pairs should return without error
  controller.Match({});

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Self-match pairs should be skipped
  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Match(pairs);

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Match({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto matches = data.database->ReadAllMatches();
  EXPECT_EQ(matches.size(), 1);
}

TEST(FeatureMatcherController, MatchSkipsExistingResults) {
  auto data = CreateTestData(3);

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Clear and match once
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  controller.Match({{id1, id2}});

  const auto matches_before = data.database->ReadAllMatches();
  const auto tvg_before = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_before.size(), 1);
  EXPECT_EQ(tvg_before.size(), 1);

  // Match same pair again — should skip since both matches and TVG exist
  controller.Match({{id1, id2}});

  const auto matches_after = data.database->ReadAllMatches();
  const auto tvg_after = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_after.size(), matches_before.size());
  EXPECT_EQ(tvg_after.size(), tvg_before.size());

  // Match with reversed pair — should also be skipped
  controller.Match({{id2, id1}});

  const auto matches_reversed = data.database->ReadAllMatches();
  const auto tvg_reversed = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_reversed.size(), matches_before.size());
  EXPECT_EQ(tvg_reversed.size(), tvg_before.size());
}

TEST(FeatureMatcherController, MatchMultiplePairs) {
  auto data = CreateTestData(4);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Match all pairs
  const auto pairs = AllPairs(data.image_ids);
  controller.Match(pairs);

  // 4 choose 2 = 6 pairs
  EXPECT_EQ(data.database->ReadAllMatches().size(), 6);
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

TEST(FeatureMatcherController, MatchPropagatesMatcherErrors) {
  auto data = CreateTestData(4);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  // Descriptors of an unexpected dimension make the SIFT matcher throw for all
  // pairs with the first image.
  std::vector<FeatureDescriptors> descriptors;
  descriptors.reserve(data.image_ids.size());
  for (const image_t image_id : data.image_ids) {
    descriptors.push_back(data.database->ReadDescriptors(image_id));
  }
  descriptors[0].data =
      FeatureDescriptorsData::Zero(descriptors[0].data.rows(), 64);
  data.database->ClearDescriptors();
  for (size_t i = 0; i < data.image_ids.size(); ++i) {
    data.database->WriteDescriptors(data.image_ids[i], descriptors[i]);
  }
  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // The error is rethrown on the calling thread instead of terminating.
  EXPECT_THAT(
      ThrownMessage([&]() { controller.Match(AllPairs(data.image_ids)); }),
      testing::HasSubstr("kSiftDescriptorDim"));
  EXPECT_FALSE(
      data.database->ExistsMatches(data.image_ids[0], data.image_ids[1]));

  // The controller remains usable for pairs without the broken image.
  controller.Match({{data.image_ids[1], data.image_ids[2]}});
  EXPECT_TRUE(
      data.database->ExistsMatches(data.image_ids[1], data.image_ids[2]));
}

TEST(FeatureMatcherController, SetupPropagatesMatcherCreationErrors) {
  auto data = CreateTestData(2);
  FeatureMatchingOptions matching_options(FeatureMatcherType::ALIKED_LIGHTGLUE);
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  matching_options.aliked->lightglue.model_path =
      (data.test_dir / "missing_model.onnx").string();
  TwoViewGeometryOptions geometry_options;
#ifdef COLMAP_ONNX_ENABLED
  const std::string expected_message = "missing_model.onnx";
#else
  const std::string expected_message = "requires ONNX support";
#endif
  {
    FeatureMatcherController controller(
        matching_options, geometry_options, data.cache);
    // The worker fails to create the matcher. Setup rethrows its error, and
    // the controller then destructs without terminating.
    EXPECT_THAT(ThrownMessage([&]() { controller.Setup(); }),
                testing::HasSubstr(expected_message));
  }

  // Also test SharedFeatureMatcher error propagation across multiple sibling
  // workers: only the creator worker rethrows on Wait(), while sibling workers
  // signal invalid setup and join without throwing.
  {
    JobQueue<FeatureMatcherData> input_queue;
    JobQueue<FeatureMatcherData> output_queue;
    auto shared_matcher = std::make_shared<SharedFeatureMatcher>();
    std::vector<std::unique_ptr<FeatureMatcherWorker>> workers;
    workers.reserve(3);
    for (int w = 0; w < 3; ++w) {
      workers.push_back(
          std::make_unique<FeatureMatcherWorker>(matching_options,
                                                 geometry_options,
                                                 data.cache,
                                                 &input_queue,
                                                 &output_queue,
                                                 shared_matcher,
                                                 /*is_shared_creator=*/w == 0));
    }
    for (auto& worker : workers) {
      worker->Start();
    }
    EXPECT_FALSE(workers[0]->CheckValidSetup());
    EXPECT_THAT(ThrownMessage([&]() { workers[0]->Wait(); }),
                testing::HasSubstr(expected_message));
    for (size_t w = 1; w < workers.size(); ++w) {
      EXPECT_FALSE(workers[w]->CheckValidSetup());
      EXPECT_NO_THROW(workers[w]->Wait());
    }
  }

  // If the creator fails in Run() before reaching SharedFeatureMatcher::Create
  // (e.g., while reading MaxNumKeypoints from a closed database), Fail() must
  // still unblock sibling workers waiting in SharedFeatureMatcher::Wait().
  {
    auto closed_db = Database::Open(data.test_dir / "closed.db");
    closed_db->Close();
    auto closed_cache = std::make_shared<FeatureMatcherCache>(100, closed_db);
    JobQueue<FeatureMatcherData> input_queue;
    JobQueue<FeatureMatcherData> output_queue;
    auto shared_matcher = std::make_shared<SharedFeatureMatcher>();
    FeatureMatchingOptions valid_options = DefaultMatchingOptions();
    std::vector<std::unique_ptr<FeatureMatcherWorker>> workers;
    workers.reserve(3);
    for (int w = 0; w < 3; ++w) {
      workers.push_back(std::make_unique<FeatureMatcherWorker>(
          valid_options,
          geometry_options,
          w == 0 ? closed_cache : data.cache,
          &input_queue,
          &output_queue,
          shared_matcher,
          /*is_shared_creator=*/w == 0));
    }
    for (size_t w = 1; w < workers.size(); ++w) {
      workers[w]->Start();
    }
    workers[0]->Start();
    EXPECT_FALSE(workers[0]->CheckValidSetup());
    EXPECT_THAT(ThrownMessage([&]() { workers[0]->Wait(); }),
                testing::HasSubstr("database"));
    for (size_t w = 1; w < workers.size(); ++w) {
      EXPECT_FALSE(workers[w]->CheckValidSetup());
      EXPECT_NO_THROW(workers[w]->Wait());
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  if (GetNumCudaDevices() > 0) {
    FeatureMatchingOptions flexglue_options(FeatureMatcherType::FLEXGLUE);
    flexglue_options.use_gpu = true;
    flexglue_options.gpu_index = "0";
    flexglue_options.num_threads = 2;
    flexglue_options.flexglue->image_path = data.test_dir.string();
    flexglue_options.flexglue->num_workers_per_gpu = 3;
    flexglue_options.flexglue->descriptor_model_path =
        (data.test_dir / "missing_model.onnx").string();
    FeatureMatcherController controller(
        flexglue_options, geometry_options, data.cache);
    EXPECT_THAT(ThrownMessage([&]() { controller.Setup(); }),
                testing::HasSubstr(expected_message));
  }
#endif
}

TEST(FeatureMatcherController, SharedFeatureMatcherWorkersMatchPairs) {
  auto data = CreateTestData(4);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  std::vector<FeatureDescriptors> normalized_descriptors;
  normalized_descriptors.reserve(data.image_ids.size());
  for (const image_t image_id : data.image_ids) {
    FeatureDescriptors desc = data.database->ReadDescriptors(image_id);
    FeatureDescriptorsFloatData float_desc = desc.ToFloat().data;
    L2NormalizeFeatureDescriptors(&float_desc);
    desc.data = FeatureDescriptorsToUnsignedByte(float_desc);
    normalized_descriptors.push_back(std::move(desc));
  }
  data.database->ClearDescriptors();
  for (size_t i = 0; i < data.image_ids.size(); ++i) {
    data.database->WriteDescriptors(data.image_ids[i],
                                    normalized_descriptors[i]);
  }
  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  // Brute-force SIFT avoids SiftCPUFeatureMatcher's non-thread-safe index
  // cache so the shared matcher is safe across concurrent workers.
  matching_options.sift->cpu_brute_force_matcher = true;
  TwoViewGeometryOptions geometry_options;
  JobQueue<FeatureMatcherData> input_queue;
  JobQueue<FeatureMatcherData> output_queue;
  auto shared_matcher = std::make_shared<SharedFeatureMatcher>();

  std::vector<std::unique_ptr<FeatureMatcherWorker>> workers;
  workers.reserve(3);
  for (int w = 0; w < 3; ++w) {
    workers.push_back(
        std::make_unique<FeatureMatcherWorker>(matching_options,
                                               geometry_options,
                                               data.cache,
                                               &input_queue,
                                               &output_queue,
                                               shared_matcher,
                                               /*is_shared_creator=*/w == 0));
    workers.back()->Start();
  }
  for (auto& worker : workers) {
    ASSERT_TRUE(worker->CheckValidSetup());
  }

  const auto pairs = AllPairs(data.image_ids);
  for (const auto& [id1, id2] : pairs) {
    FeatureMatcherData job;
    job.image_id1 = id1;
    job.image_id2 = id2;
    ASSERT_TRUE(input_queue.Push(std::move(job)));
  }

  for (size_t i = 0; i < pairs.size(); ++i) {
    auto out = output_queue.Pop();
    ASSERT_TRUE(out.IsValid());
    EXPECT_EQ(out.Data().error, nullptr);
    EXPECT_FALSE(out.Data().matches.empty());
  }

  input_queue.Wait();
  for (auto& worker : workers) {
    worker->Stop();
  }
  input_queue.Stop();
  output_queue.Stop();
  for (auto& worker : workers) {
    worker->Wait();
  }

  const auto created_matcher = shared_matcher->Wait();
  EXPECT_NE(created_matcher, nullptr);
  EXPECT_THAT(
      ThrownMessage([&]() { shared_matcher->Create(matching_options); }),
      testing::HasSubstr("!done_"));
  EXPECT_EQ(shared_matcher->Wait(), created_matcher);
}

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_ONNX_ENABLED) && \
    defined(COLMAP_DOWNLOAD_ENABLED)
TEST(FeatureMatcherController, FlexGlueSharedCudaWorkersMatchPairs) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }

  constexpr int kNumImages = 4;
  constexpr int kWidth = 640;
  constexpr int kHeight = 480;
  constexpr int kStepX = 15;
  constexpr int kStepY = 10;

  const std::filesystem::path test_dir = CreateTestDir();
  const auto database = Database::Open(test_dir / "database.db");

  // Deterministic textured canvas cropped at translated offsets so all pairs
  // are covisible with known correspondences and no descriptors are stored.
  std::mt19937 prng(42);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> size_dist(8, 80);
  const int canvas_width = kWidth + (kNumImages - 1) * kStepX + 1;
  const int canvas_height = kHeight + (kNumImages - 1) * kStepY + 1;
  std::uniform_int_distribution<int> x_dist(0, canvas_width - 1);
  std::uniform_int_distribution<int> y_dist(0, canvas_height - 1);
  Bitmap canvas(canvas_width, canvas_height, /*as_rgb=*/true);
  for (int y = 0; y < canvas_height; ++y) {
    for (int x = 0; x < canvas_width; ++x) {
      canvas.SetPixel(
          x,
          y,
          BitmapColor<uint8_t>(static_cast<uint8_t>(x * 255 / canvas_width),
                               static_cast<uint8_t>(y * 255 / canvas_height),
                               128));
    }
  }
  for (int r = 0; r < 300; ++r) {
    const int size = size_dist(prng);
    const int x0 = x_dist(prng);
    const int y0 = y_dist(prng);
    const BitmapColor<uint8_t> color(static_cast<uint8_t>(byte_dist(prng)),
                                     static_cast<uint8_t>(byte_dist(prng)),
                                     static_cast<uint8_t>(byte_dist(prng)));
    for (int y = y0; y < std::min(canvas_height, y0 + size); ++y) {
      for (int x = x0; x < std::min(canvas_width, x0 + size); ++x) {
        canvas.SetPixel(x, y, color);
      }
    }
  }

  Camera camera = Camera::CreateFromModelId(
      kInvalidCameraId, PinholeCameraModel::model_id, 500, kWidth, kHeight);
  camera.camera_id = database->WriteCamera(camera);

  constexpr int kGridRows = 16;
  constexpr int kGridCols = 20;
  std::vector<image_t> image_ids;
  image_ids.reserve(kNumImages);
  for (int i = 0; i < kNumImages; ++i) {
    Bitmap crop(kWidth, kHeight, /*as_rgb=*/true);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        crop.SetPixel(x, y, *canvas.GetPixel(i * kStepX + x, i * kStepY + y));
      }
    }
    const std::string name = "image" + std::to_string(i) + ".png";
    ASSERT_TRUE(crop.Write(test_dir / name));

    Image image;
    image.SetName(name);
    image.SetCameraId(camera.camera_id);
    const image_t image_id = database->WriteImage(image);
    image_ids.push_back(image_id);

    FeatureKeypoints keypoints;
    keypoints.reserve(kGridRows * kGridCols);
    for (int r = 0; r < kGridRows; ++r) {
      for (int c = 0; c < kGridCols; ++c) {
        keypoints.emplace_back(80 + c * 24 - i * kStepX + 0.5f,
                               60 + r * 22 - i * kStepY + 0.5f);
      }
    }
    database->WriteKeypoints(image_id, keypoints);
  }

  const auto pairs = AllPairs(image_ids);
  auto run_with_workers = [&](int num_workers_per_gpu) {
    database->ClearMatches();
    database->ClearTwoViewGeometries();
    auto cache = std::make_shared<FeatureMatcherCache>(100, database);
    FeatureMatchingOptions matching_options(FeatureMatcherType::FLEXGLUE);
    matching_options.use_gpu = true;
    matching_options.gpu_index = "0";
    // Smaller than num_workers_per_gpu = 3 to also exercise the std::max(1, .)
    // clamping when dividing num_threads across GPU workers.
    matching_options.num_threads = 2;
    matching_options.skip_geometric_verification = true;
    matching_options.flexglue->image_path = test_dir.string();
    matching_options.flexglue->num_workers_per_gpu = num_workers_per_gpu;
    TwoViewGeometryOptions geometry_options;
    FeatureMatcherController controller(
        matching_options, geometry_options, cache);
    EXPECT_TRUE(controller.Setup());
    controller.Match(pairs);

    std::vector<FeatureMatches> all_matches;
    all_matches.reserve(pairs.size());
    for (const auto& [id1, id2] : pairs) {
      all_matches.push_back(database->ReadMatches(id1, id2));
    }
    return all_matches;
  };

  const std::vector<FeatureMatches> expected_matches = run_with_workers(1);
  for (const FeatureMatches& matches : expected_matches) {
    ASSERT_FALSE(matches.empty());
  }

  const std::vector<FeatureMatches> concurrent_matches = run_with_workers(3);
  EXPECT_EQ(concurrent_matches, expected_matches);
}
#endif

TEST(FeatureMatcherController, MatchSkipGeometricVerification) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  controller.Match({{data.image_ids[0], data.image_ids[1]}});

  // Matches should be written even without geometric verification
  EXPECT_EQ(data.database->ReadAllMatches().size(), 1);

  // Geometric verification was skipped: no TVG row is stored at all.
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(data.image_ids[0],
                                                    data.image_ids[1]));
}

TEST(FeatureMatcherController, MatchSkipGeometricVerificationTwice) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const std::vector<std::pair<image_t, image_t>> pairs = {
      {data.image_ids[0], data.image_ids[1]}};
  controller.Match(pairs);

  // Rematching pairs with existing matches must be a no-op. Without stored
  // two-view geometries, there is nothing to verify and no verifier workers
  // exist to drain the verifier queue.
  controller.Match(pairs);

  EXPECT_EQ(data.database->ReadAllMatches().size(), 1);
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(data.image_ids[0],
                                                    data.image_ids[1]));
}

TEST(GeometricVerifierController, OptionsAccessor) {
  auto data = CreateTestData(3);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);

  EXPECT_EQ(controller.Options().num_threads, 1);
  controller.Options().num_threads = 2;
  EXPECT_EQ(controller.Options().num_threads, 2);
}

TEST(GeometricVerifierController, VerifyEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Verifying empty pairs should return without error
  controller.Verify({});

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Verify(pairs);

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 2);
  MatchPairsWithoutVerification(data, {{data.image_ids[0], data.image_ids[1]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Verify({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto tvgs = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(tvgs.size(), 1);
}

TEST(GeometricVerifierController, VerifyWithExistingMatches) {
  auto data = CreateTestData(4);
  const auto pairs = AllPairs(data.image_ids);
  MatchPairsWithoutVerification(data, pairs);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify(pairs);

  // All 6 pairs should now have TVGs
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

TEST(GeometricVerifierController, VerifyPropagatesErrors) {
  auto data = CreateTestData(3);
  const auto pairs = AllPairs(data.image_ids);
  MatchPairsWithoutVerification(data, pairs);

  // Keypoints in an unsupported format make reading them throw for all pairs
  // with the first image.
  std::vector<FeatureKeypointsBlob> keypoints;
  keypoints.reserve(data.image_ids.size());
  for (const image_t image_id : data.image_ids) {
    keypoints.push_back(data.database->ReadKeypointsBlob(image_id));
  }
  keypoints[0] = FeatureKeypointsBlob::Zero(keypoints[0].rows(), 3);
  data.database->ClearKeypoints();
  for (size_t i = 0; i < data.image_ids.size(); ++i) {
    data.database->WriteKeypoints(data.image_ids[i], keypoints[i]);
  }
  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;
  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // The error is rethrown instead of the verifier thread exiting and the
  // controller waiting forever for its output. The other pairs are verified.
  EXPECT_THAT(ThrownMessage([&]() { controller.Verify(pairs); }),
              testing::HasSubstr("Keypoint format not supported"));
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(data.image_ids[0],
                                                    data.image_ids[1]));
  EXPECT_TRUE(data.database->ExistsTwoViewGeometry(data.image_ids[1],
                                                   data.image_ids[2]));
}

TEST(GeometricVerifierController, VerifyClearsDegenerateGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  GeometricVerifierController controller(
      verifier_options, LowRatioGeometryOptions(), data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{pair_data.image_id1, pair_data.image_id2}});

  // The ratio recheck rejects the pair: the DEGENERATE diagnosis is kept
  // but nothing else may be stored.
  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  // Raw matches are preserved.
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

TEST(FeatureMatcherController, MatchClearsDegenerateGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  FeatureMatcherController controller(
      matching_options, LowRatioGeometryOptions(), data.cache);
  ASSERT_TRUE(controller.Setup());

  // Matches exist without TVG, so Match takes the verification leg.
  controller.Match({{pair_data.image_id1, pair_data.image_id2}});

  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

TEST(FeatureMatcherController,
     MatchSkippedPairsStoreDegenerateTwoViewGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;
  // No descriptors and no pre-existing matches: matching is skipped.
  data.database->ClearMatches();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Match({{pair_data.image_id1, pair_data.image_id2}});

  // Skipped pairs are labeled DEGENERATE but store no payload.
  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  EXPECT_TRUE(
      data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
          .empty());
}

TEST(GeometricVerifierController,
     VerifySkippedPairsStoreDegenerateTwoViewGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 200;  // Above the 100 stored matches.
  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{pair_data.image_id1, pair_data.image_id2}});

  // Below-minimum pairs are labeled DEGENERATE but store no payload.
  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  // Raw matches are preserved.
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

}  // namespace
}  // namespace colmap
