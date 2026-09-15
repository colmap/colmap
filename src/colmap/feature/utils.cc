// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/utils.h"

#include "colmap/math/math.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace colmap {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

void L2NormalizeFeatureDescriptors(FeatureDescriptorsFloatData* descriptors) {
  descriptors->rowwise().normalize();
}

void L1RootNormalizeFeatureDescriptors(
    FeatureDescriptorsFloatData* descriptors) {
  for (Eigen::Index r = 0; r < descriptors->rows(); ++r) {
    descriptors->row(r) *= 1 / descriptors->row(r).lpNorm<1>();
    descriptors->row(r) = descriptors->row(r).array().sqrt();
  }
}

FeatureDescriptorsData FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloatData>& descriptors) {
  FeatureDescriptorsData descriptors_unsigned_byte(descriptors.rows(),
                                                   descriptors.cols());
  for (Eigen::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features) {
  THROW_CHECK_EQ(keypoints->size(), descriptors->data.rows());
  THROW_CHECK_GT(num_features, 0);

  if (static_cast<size_t>(descriptors->data.rows()) <= num_features) {
    return;
  }

  std::vector<std::pair<size_t, float>> scales;
  scales.reserve(keypoints->size());
  for (size_t i = 0; i < keypoints->size(); ++i) {
    scales.emplace_back(i, (*keypoints)[i].ComputeScale());
  }

  std::partial_sort(scales.begin(),
                    scales.begin() + num_features,
                    scales.end(),
                    [](const std::pair<size_t, float>& scale1,
                       const std::pair<size_t, float>& scale2) {
                      return scale1.second > scale2.second;
                    });

  FeatureKeypoints top_scale_keypoints(num_features);
  FeatureDescriptors top_scale_descriptors;
  top_scale_descriptors.data.resize(num_features, descriptors->data.cols());
  top_scale_descriptors.type = descriptors->type;
  for (size_t i = 0; i < num_features; ++i) {
    top_scale_keypoints[i] = (*keypoints)[scales[i].first];
    top_scale_descriptors.data.row(i) = descriptors->data.row(scales[i].first);
  }

  *keypoints = std::move(top_scale_keypoints);
  *descriptors = std::move(top_scale_descriptors);
}

std::vector<float> HWCToCHW(const uint8_t* data,
                            int width,
                            int height,
                            int pitch) {
  THROW_CHECK_NOTNULL(data);
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  THROW_CHECK_GE(pitch, 3 * width);

  const int num_pixels = width * height;
  std::vector<float> chw(static_cast<size_t>(3) * num_pixels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        chw[c * num_pixels + y * width + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }
  return chw;
}

std::vector<float> BitmapToCHW(const Bitmap& bitmap) {
  THROW_CHECK(bitmap.IsRGB());
  return HWCToCHW(bitmap.RowMajorData().data(),
                  bitmap.Width(),
                  bitmap.Height(),
                  bitmap.Pitch());
}

bool CheckDetectionOptions(int max_num_features, double min_score) {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_IN_RANGE(min_score, 0, 1);
  return true;
}

bool CheckGPUOptions(bool use_gpu,
                     const std::string& gpu_index,
                     const char* feature_kind) {
  if (!use_gpu) {
    return true;
  }
  CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
#ifndef COLMAP_GPU_ENABLED
  LOG(ERROR) << "Cannot use GPU " << feature_kind
             << " without CUDA or OpenGL support. "
                "Consider setting use_gpu to false.";
  return false;
#endif
  return true;
}

}  // namespace colmap
