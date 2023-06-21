#include "colmap/feature/torch.h"

#include "colmap/feature/sift.h"

#include "lib/VLFeat/covdet.h"

#include <memory>

#include <torch/script.h>

namespace colmap {

bool ExtractCovariantFeatures(const TorchFeatureOptions& options,
                              const Bitmap& bitmap,
                              FeatureKeypoints* keypoints,
                              FeatureDescriptors* descriptors) {
  CHECK(options.sift_options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  std::cout << "TORCH_FEATURE" << std::endl;

  torch::jit::script::Module feature_extractor;
  try {
    feature_extractor = torch::jit::load(options.torch_model_path);
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: Failed to load torch model: " << e.what() << std::endl;
    return false;
  }

  // Setup covariant SIFT detector.
  std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
  if (!covdet) {
    return false;
  }

  constexpr int kMaxOctaveResolution = 1000;
  CHECK_LE(options.sift_options.octave_resolution, kMaxOctaveResolution);

  vl_covdet_set_first_octave(covdet.get(), options.sift_options.first_octave);
  vl_covdet_set_octave_resolution(covdet.get(),
                                  options.sift_options.octave_resolution);
  vl_covdet_set_peak_threshold(covdet.get(),
                               options.sift_options.peak_threshold);
  vl_covdet_set_edge_threshold(covdet.get(),
                               options.sift_options.edge_threshold);

  {
    const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
    std::vector<float> data_float(data_uint8.size());
    for (size_t i = 0; i < data_uint8.size(); ++i) {
      data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
    vl_covdet_put_image(
        covdet.get(), data_float.data(), bitmap.Width(), bitmap.Height());
  }

  vl_covdet_detect(covdet.get(), options.sift_options.max_num_features);

  if (!options.sift_options.upright) {
    if (options.sift_options.estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }
  }

  const int num_features = vl_covdet_get_num_features(covdet.get());
  VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

  // Sort features according to detected octave and scale.
  std::sort(
      features,
      features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        } else {
          return feature1.o > feature2.o;
        }
      });

  const size_t max_num_features =
      static_cast<size_t>(options.sift_options.max_num_features);

  // Copy detected keypoints and clamp when maximum number of features reached.
  int prev_octave_scale_idx = std::numeric_limits<int>::max();
  for (int i = 0; i < num_features; ++i) {
    FeatureKeypoint keypoint;
    keypoint.x = features[i].frame.x + 0.5;
    keypoint.y = features[i].frame.y + 0.5;
    keypoint.a11 = features[i].frame.a11;
    keypoint.a12 = features[i].frame.a12;
    keypoint.a21 = features[i].frame.a21;
    keypoint.a22 = features[i].frame.a22;
    keypoints->push_back(keypoint);

    const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
    CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

    if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints->size() >= max_num_features) {
      break;
    }

    prev_octave_scale_idx = octave_scale_idx;
  }

  constexpr int kPatchSize = 32;
  constexpr int kVlfeatPatchExtent = 16;
  constexpr int kVlfeatPatchSize = 2 * kVlfeatPatchExtent + 1;
  constexpr double kVlfeatPatchRelativeExtent = 7.5;
  constexpr double kVlfeatPatchRelativeSmoothing = 1;

  const int num_keypoints = keypoints->size();
  at::Tensor torch_patches =
      at::zeros({num_keypoints, 1, kPatchSize, kPatchSize});
  std::vector<float> vlfeat_patch(kVlfeatPatchSize * kVlfeatPatchSize);
  std::vector<float> patch(kPatchSize * kPatchSize);
  for (int i = 0; i < num_keypoints; ++i) {
    vl_covdet_extract_patch_for_frame(covdet.get(),
                                      vlfeat_patch.data(),
                                      kVlfeatPatchExtent,
                                      kVlfeatPatchRelativeExtent,
                                      kVlfeatPatchRelativeSmoothing,
                                      features[i].frame);

    const float weight_right = std::fmod((*keypoints)[i].x, 1);
    const float weight_bottom = std::fmod((*keypoints)[i].y, 1);
    const float weight_left = 1.f - weight_right;
    const float weight_top = 1.f - weight_bottom;
    for (int y = 0; y < kPatchSize; ++y) {
      const float* top_line = &vlfeat_patch[y * kPatchSize];
      const float* bottom_line = &vlfeat_patch[(y + 1) * kPatchSize];
      for (int x = 0; x < kPatchSize; ++x) {
        const float top =
            weight_left * top_line[x] + weight_right * top_line[x + 1];
        const float bottom =
            weight_left * bottom_line[x] + weight_right * bottom_line[x + 1];
        torch_patches[i][0][y][x] = weight_top * top + weight_bottom * bottom;
      }
    }
  }

  std::vector<torch::jit::IValue> inputs = {torch_patches};
  at::Tensor torch_descriptors = feature_extractor.forward(inputs).toTensor();
  auto torch_descriptors_access = torch_descriptors.accessor<float, 2>();

  descriptors->resize(num_keypoints, 128);
  for (int i = 0; i < num_keypoints; ++i) {
    for (int j = 0; j < 128; ++j) {
      (*descriptors)(i, j) = std::min(
          std::max(255.f * torch_descriptors_access[i][j], 0.f), 255.f);
    }
  }

  return true;
}

}  // namespace colmap
