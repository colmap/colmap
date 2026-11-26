// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/apriltag.h"

#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#ifdef COLMAP_APRILTAG_ENABLED
#include <apriltag/apriltag.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tagCircle21h7.h>
#include <apriltag/tagStandard41h12.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_APRILTAG_ENABLED

struct FamilyFactory {
  apriltag_family_t* (*create)();
  void (*destroy)(apriltag_family_t*);
};

FamilyFactory GetFamilyFactory(const AprilTagFamily family) {
  switch (family) {
    case AprilTagFamily::TAG_36H11:
      return {&tag36h11_create, &tag36h11_destroy};
    case AprilTagFamily::TAG_25H9:
      return {&tag25h9_create, &tag25h9_destroy};
    case AprilTagFamily::TAG_16H5:
      return {&tag16h5_create, &tag16h5_destroy};
    case AprilTagFamily::TAG_36H10:
      return {&tag36h10_create, &tag36h10_destroy};
    case AprilTagFamily::TAG_STANDARD_41H12:
      return {&tagStandard41h12_create, &tagStandard41h12_destroy};
    case AprilTagFamily::TAG_CIRCLE_21H7:
      return {&tagCircle21h7_create, &tagCircle21h7_destroy};
  }
  throw std::runtime_error("Unsupported AprilTag family.");
}

using FamilyPtr =
    std::unique_ptr<apriltag_family_t, void (*)(apriltag_family_t*)>;
using DetectorPtr =
    std::unique_ptr<apriltag_detector_t, void (*)(apriltag_detector_t*)>;
using ImageU8Ptr = std::unique_ptr<image_u8_t, void (*)(image_u8_t*)>;
using ZArrayPtr = std::unique_ptr<zarray_t, void (*)(zarray_t*)>;

#endif

}  // namespace

AprilTagFamily ParseAprilTagFamily(const std::string_view name) {
  std::string normalized(name);
  StringToUpper(&normalized);
  if (normalized == "TAG_36H11" || normalized == "TAG36H11") {
    return AprilTagFamily::TAG_36H11;
  }
  if (normalized == "TAG_25H9" || normalized == "TAG25H9") {
    return AprilTagFamily::TAG_25H9;
  }
  if (normalized == "TAG_16H5" || normalized == "TAG16H5") {
    return AprilTagFamily::TAG_16H5;
  }
  if (normalized == "TAG_36H10" || normalized == "TAG36H10") {
    return AprilTagFamily::TAG_36H10;
  }
  if (normalized == "TAG_STANDARD_41H12" || normalized == "TAGSTANDARD41H12") {
    return AprilTagFamily::TAG_STANDARD_41H12;
  }
  if (normalized == "TAG_CIRCLE_21H7" || normalized == "TAGCIRCLE21H7") {
    return AprilTagFamily::TAG_CIRCLE_21H7;
  }
  throw std::runtime_error("Unknown AprilTag family: " + std::string(name));
}

bool AprilTagDetectionOptions::Check() const {
  CHECK_OPTION_GE(quad_decimate, 1.0);
  CHECK_OPTION(quad_decimate == 1.5 ||
               std::floor(quad_decimate) == quad_decimate);
  CHECK_OPTION(std::isfinite(quad_sigma));
  CHECK_OPTION(std::isfinite(decode_sharpening));
  CHECK_OPTION_GE(max_hamming, 0);
  CHECK_OPTION_LE(max_hamming, 2);
  CHECK_OPTION_GE(min_decision_margin, 0.0);
  CHECK_OPTION(num_threads == -1 || num_threads > 0);
  return true;
}

#ifdef COLMAP_APRILTAG_ENABLED

struct AprilTagDetector::Impl {
  explicit Impl(const AprilTagDetectionOptions& options)
      : factory(GetFamilyFactory(options.family)),
        family(factory.create(), factory.destroy),
        detector(apriltag_detector_create(), &apriltag_detector_destroy) {
    THROW_CHECK(family != nullptr);
    THROW_CHECK(detector != nullptr);

    detector->quad_decimate = static_cast<float>(options.quad_decimate);
    detector->quad_sigma = static_cast<float>(options.quad_sigma);
    detector->refine_edges = options.refine_edges;
    detector->decode_sharpening = options.decode_sharpening;
    detector->nthreads = GetEffectiveNumThreads(options.num_threads);

    apriltag_detector_add_family_bits(
        detector.get(),
        family.get(),
        options.max_hamming);
  }

  ~Impl() {
    // Clear family registration before the family object is destroyed.
    if (detector != nullptr) {
      apriltag_detector_clear_families(detector.get());
    }
  }

  FamilyFactory factory;
  FamilyPtr family;
  DetectorPtr detector;
};

AprilTagDetector::AprilTagDetector(const AprilTagDetectionOptions& options)
    : options_(options) {
  THROW_CHECK(options_.Check());
  impl_ = std::make_unique<Impl>(options_);
}

AprilTagDetector::~AprilTagDetector() = default;
AprilTagDetector::AprilTagDetector(AprilTagDetector&&) noexcept = default;
AprilTagDetector& AprilTagDetector::operator=(AprilTagDetector&&) noexcept =
    default;

std::vector<AprilTagDetection> AprilTagDetector::Detect(
    const Bitmap& bitmap) const {
  if (bitmap.IsEmpty() || bitmap.Width() < 8 || bitmap.Height() < 8) {
    return {};
  }

  std::optional<Bitmap> grey_storage;
  const Bitmap* grey_bitmap = &bitmap;
  if (!bitmap.IsGrey()) {
    grey_storage.emplace(bitmap.CloneAsGrey());
    grey_bitmap = &*grey_storage;
  }

  const int width = grey_bitmap->Width();
  const int height = grey_bitmap->Height();
  const uint8_t* src_data = grey_bitmap->RowMajorData().data();

  // apriltag_detector_detect blurs/sharpens im_orig in place when
  // quad_decimate <= 1.0 and quad_sigma != 0.0; copy only in that case.
  ImageU8Ptr owned_image(nullptr, &image_u8_destroy);
  image_u8_t borrowed_image{
      width,
      height,
      width,
      const_cast<uint8_t*>(src_data),
  };
  image_u8_t* im_ptr = &borrowed_image;
  if (impl_->detector->quad_decimate <= 1.0f &&
      impl_->detector->quad_sigma != 0.0f) {
    owned_image.reset(image_u8_create(width, height));
    THROW_CHECK(owned_image != nullptr);
    for (int y = 0; y < height; ++y) {
      std::memcpy(owned_image->buf + y * owned_image->stride,
                  src_data + static_cast<size_t>(y) * width,
                  width);
    }
    im_ptr = owned_image.get();
  }

  ZArrayPtr raw_detections(
      apriltag_detector_detect(impl_->detector.get(), im_ptr),
      &apriltag_detections_destroy);
  if (raw_detections == nullptr) {
    return {};
  }

  std::vector<AprilTagDetection> detections;
  const int num_detections = zarray_size(raw_detections.get());
  detections.reserve(num_detections);
  for (int i = 0; i < num_detections; ++i) {
    apriltag_detection_t* det = nullptr;
    zarray_get(raw_detections.get(), i, &det);
    if (det == nullptr) {
      continue;
    }
    if (det->hamming > options_.max_hamming ||
        static_cast<double>(det->decision_margin) <
            options_.min_decision_margin) {
      continue;
    }

    AprilTagDetection detection;
    detection.id = det->id;
    detection.center = Eigen::Vector2d(det->c[0], det->c[1]);
    for (size_t corner_idx = 0; corner_idx < 4; ++corner_idx) {
      detection.corners[corner_idx] =
          Eigen::Vector2d(det->p[corner_idx][0], det->p[corner_idx][1]);
    }
    detection.decision_margin = static_cast<double>(det->decision_margin);
    detection.hamming = det->hamming;
    detections.push_back(detection);
  }

  std::sort(detections.begin(),
            detections.end(),
            [](const AprilTagDetection& a, const AprilTagDetection& b) {
              if (a.decision_margin != b.decision_margin) {
                return a.decision_margin > b.decision_margin;
              }
              if (a.id != b.id) {
                return a.id < b.id;
              }
              if (a.center.x() != b.center.x()) {
                return a.center.x() < b.center.x();
              }
              return a.center.y() < b.center.y();
            });

  return detections;
}

std::vector<AprilTagDetection> DetectAprilTags(
    const Bitmap& bitmap, const AprilTagDetectionOptions& options) {
  AprilTagDetector detector(options);
  return detector.Detect(bitmap);
}

Bitmap SynthesizeAprilTagBitmap(const AprilTagFamily family,
                                const int tag_id,
                                const int quiet_zone_bits,
                                const int pixel_scale) {
  THROW_CHECK_GE(tag_id, 0);
  THROW_CHECK_GE(quiet_zone_bits, 0);
  THROW_CHECK_GE(pixel_scale, 1);

  const FamilyFactory factory = GetFamilyFactory(family);
  const FamilyPtr fam(factory.create(), factory.destroy);
  THROW_CHECK(fam != nullptr);
  THROW_CHECK_LT(static_cast<uint32_t>(tag_id), fam->ncodes);

  const ImageU8Ptr tag_image(
      apriltag_to_image(fam.get(), static_cast<uint32_t>(tag_id)),
      &image_u8_destroy);
  THROW_CHECK(tag_image != nullptr);

  const int total_width_bits = tag_image->width + 2 * quiet_zone_bits;
  const int total_height_bits = tag_image->height + 2 * quiet_zone_bits;
  Bitmap bitmap(total_width_bits * pixel_scale,
                total_height_bits * pixel_scale,
                /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(255));

  std::vector<uint8_t>& data = bitmap.RowMajorData();
  const int out_width = bitmap.Width();
  for (int y = 0; y < tag_image->height; ++y) {
    const int dst_y0 = (y + quiet_zone_bits) * pixel_scale;
    for (int x = 0; x < tag_image->width; ++x) {
      const uint8_t value = tag_image->buf[y * tag_image->stride + x];
      if (value == 255) {
        continue;
      }
      const int dst_x0 = (x + quiet_zone_bits) * pixel_scale;
      for (int dy = 0; dy < pixel_scale; ++dy) {
        std::memset(&data[(dst_y0 + dy) * out_width + dst_x0],
                    value,
                    static_cast<size_t>(pixel_scale));
      }
    }
  }
  return bitmap;
}

#else

struct AprilTagDetector::Impl {};

AprilTagDetector::AprilTagDetector(const AprilTagDetectionOptions& options)
    : options_(options) {
  throw std::runtime_error("AprilTag support is not enabled in this build.");
}

AprilTagDetector::~AprilTagDetector() = default;
AprilTagDetector::AprilTagDetector(AprilTagDetector&&) noexcept = default;
AprilTagDetector& AprilTagDetector::operator=(AprilTagDetector&&) noexcept =
    default;

std::vector<AprilTagDetection> AprilTagDetector::Detect(
    const Bitmap& /*bitmap*/) const {
  throw std::runtime_error("AprilTag support is not enabled in this build.");
}

std::vector<AprilTagDetection> DetectAprilTags(
    const Bitmap& /*bitmap*/, const AprilTagDetectionOptions& /*options*/) {
  throw std::runtime_error("AprilTag support is not enabled in this build.");
}

Bitmap SynthesizeAprilTagBitmap(AprilTagFamily /*family*/,
                                int /*tag_id*/,
                                int /*quiet_zone_bits*/,
                                int /*pixel_scale*/) {
  throw std::runtime_error("AprilTag support is not enabled in this build.");
}

#endif

}  // namespace colmap
