// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/apriltag.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AprilTag, ParseFamily) {
  EXPECT_EQ(ParseAprilTagFamily("TAG_36H11"), AprilTagFamily::TAG_36H11);
  EXPECT_EQ(ParseAprilTagFamily("tag36h11"), AprilTagFamily::TAG_36H11);
  EXPECT_EQ(ParseAprilTagFamily("tag25h9"), AprilTagFamily::TAG_25H9);
  EXPECT_EQ(ParseAprilTagFamily("tag16h5"), AprilTagFamily::TAG_16H5);
  EXPECT_EQ(ParseAprilTagFamily("tag36h10"), AprilTagFamily::TAG_36H10);
  EXPECT_EQ(ParseAprilTagFamily("tagStandard41h12"),
            AprilTagFamily::TAG_STANDARD_41H12);
  EXPECT_EQ(ParseAprilTagFamily("tagCircle21h7"),
            AprilTagFamily::TAG_CIRCLE_21H7);
  EXPECT_THROW(ParseAprilTagFamily("unknown_family"), std::runtime_error);
}

TEST(AprilTag, OptionsCheck) {
  AprilTagDetectionOptions options;
  EXPECT_TRUE(options.Check());

  options.quad_decimate = 1.0;
  EXPECT_TRUE(options.Check());
  options.quad_decimate = 1.5;
  EXPECT_TRUE(options.Check());
  options.quad_decimate = 0.5;
  EXPECT_FALSE(options.Check());
  options.quad_decimate = 2.5;
  EXPECT_FALSE(options.Check());
  options.quad_decimate = 2.0;

  options.max_hamming = -1;
  EXPECT_FALSE(options.Check());
  options.max_hamming = 3;
  EXPECT_FALSE(options.Check());
  options.max_hamming = 1;

  options.min_decision_margin = -1.0;
  EXPECT_FALSE(options.Check());
  options.min_decision_margin = 25.0;

  options.num_threads = 0;
  EXPECT_FALSE(options.Check());
  options.num_threads = 1;
  EXPECT_TRUE(options.Check());
}

#ifdef COLMAP_APRILTAG_ENABLED

TEST(AprilTag, EmptyAndBlankBitmaps) {
  AprilTagDetector detector;
  EXPECT_TRUE(detector.Detect(Bitmap()).empty());

  Bitmap tiny(4, 4, false);
  tiny.Fill(BitmapColor<uint8_t>(255));
  EXPECT_TRUE(detector.Detect(tiny).empty());

  Bitmap blank(200, 200, false);
  blank.Fill(BitmapColor<uint8_t>(200));
  EXPECT_TRUE(detector.Detect(blank).empty());
}

TEST(AprilTag, DetectAllSupportedFamilies) {
  const std::vector<std::pair<AprilTagFamily, int>> cases = {
      {AprilTagFamily::TAG_36H11, 7},
      {AprilTagFamily::TAG_25H9, 3},
      {AprilTagFamily::TAG_16H5, 5},
      {AprilTagFamily::TAG_36H10, 11},
      {AprilTagFamily::TAG_STANDARD_41H12, 4},
      {AprilTagFamily::TAG_CIRCLE_21H7, 2},
  };

  for (const auto& [family, tag_id] : cases) {
    SCOPED_TRACE(AprilTagFamilyToString(family));
    const Bitmap bitmap = SynthesizeAprilTagBitmap(
        family, tag_id, /*quiet_zone_bits=*/2, /*pixel_scale=*/20);
    AprilTagDetectionOptions options;
    options.family = family;
    options.quad_decimate = 1.0;
    const auto detections = DetectAprilTags(bitmap, options);
    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].id, tag_id);
    EXPECT_EQ(detections[0].hamming, 0);
    EXPECT_GE(detections[0].decision_margin, options.min_decision_margin);
    EXPECT_NEAR(detections[0].center.x(), bitmap.Width() / 2.0, 0.25);
    EXPECT_NEAR(detections[0].center.y(), bitmap.Height() / 2.0, 0.25);
  }
}

TEST(AprilTag, SubpixelCenterConventionAndRGBConversion) {
  // Tag is symmetrically centered in the bitmap; COLMAP places pixel (i, j)
  // center at (i + 0.5, j + 0.5) so the exact bitmap center is (W/2, H/2).
  const Bitmap grey_bitmap = SynthesizeAprilTagBitmap(AprilTagFamily::TAG_36H11,
                                                      /*tag_id=*/42,
                                                      /*quiet_zone_bits=*/3,
                                                      /*pixel_scale=*/16);
  const Bitmap rgb_bitmap = grey_bitmap.CloneAsRGB();

  for (const double quad_decimate : {1.0, 1.5, 2.0}) {
    SCOPED_TRACE(quad_decimate);
    AprilTagDetectionOptions options;
    options.family = AprilTagFamily::TAG_36H11;
    options.quad_decimate = quad_decimate;
    options.num_threads = 1;

    const auto detections_grey = DetectAprilTags(grey_bitmap, options);
    const auto detections_rgb = DetectAprilTags(rgb_bitmap, options);
    ASSERT_EQ(detections_grey.size(), 1);
    ASSERT_EQ(detections_rgb.size(), 1);
    EXPECT_EQ(detections_grey[0].id, 42);
    EXPECT_EQ(detections_rgb[0].id, 42);
    EXPECT_NEAR(detections_grey[0].center.x(),
                grey_bitmap.Width() / 2.0,
                0.25);
    EXPECT_NEAR(detections_grey[0].center.y(),
                grey_bitmap.Height() / 2.0,
                0.25);
    EXPECT_NEAR(detections_rgb[0].center.x(),
                detections_grey[0].center.x(),
                1e-6);
    EXPECT_NEAR(detections_rgb[0].center.y(),
                detections_grey[0].center.y(),
                1e-6);
  }
}

TEST(AprilTag, BlurDoesNotMutateInputBitmapAndFiltersByMargin) {
  Bitmap bitmap = SynthesizeAprilTagBitmap(AprilTagFamily::TAG_36H11,
                                           /*tag_id=*/12,
                                           /*quiet_zone_bits=*/2,
                                           /*pixel_scale=*/16);
  const std::vector<uint8_t> original_bytes = bitmap.RowMajorData();

  AprilTagDetectionOptions options;
  options.quad_decimate = 1.0;
  options.quad_sigma = 0.8;
  options.min_decision_margin = 10.0;
  const auto detections = DetectAprilTags(bitmap, options);
  ASSERT_EQ(detections.size(), 1);
  EXPECT_EQ(detections[0].id, 12);
  EXPECT_EQ(bitmap.RowMajorData(), original_bytes);

  AprilTagDetectionOptions strict_options = options;
  strict_options.min_decision_margin = 10000.0;
  EXPECT_TRUE(DetectAprilTags(bitmap, strict_options).empty());
}

#endif

}  // namespace
}  // namespace colmap
