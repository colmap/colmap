#include "colmap/feature/aliked.h"

#include "colmap/sensor/bitmap.h"

#include <iostream>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateImageWithSquare(const int size, Bitmap* bitmap) {
  bitmap->Allocate(size, size, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
}

TEST(ALIKED, NOMINAL) {
  Bitmap image;
  CreateImageWithSquare(200, &image);

  auto extractor = CreateALIKEDFeatureExtractor(ALIKEDFeatureOptions());
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));

  LOG(INFO) << descriptors->rows() << " " << descriptors->cols();

  auto matcher = CreateALIKEDLightGlueFeatureMatcher();
  FeatureMatches matches;
  matcher->Match(
      {/*image_id=*/1, image.Width(), image.Height(), descriptors, keypoints},
      {/*image_id=*/2, image.Width(), image.Height(), descriptors, keypoints},
      &matches);

  LOG(INFO) << matches.size();
}

}  // namespace
}  // namespace colmap
