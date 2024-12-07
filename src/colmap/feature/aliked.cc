#include "colmap/feature/aliked.h"

#include "thirdparty/ALIKED/aliked.hpp"

#include <memory>

namespace colmap {
namespace {

std::string_view GetDeviceName() {
#ifdef COLMAP_CUDA_ENABLED
  return "cuda";
#else
  return "cpu";
#endif
}

class ALIKEDFeatureExtractor : public FeatureExtractor {
 public:
  explicit ALIKEDFeatureExtractor(const ALIKEDFeatureOptions& options)
      : options_(options), aliked_("aliked-n32", GetDeviceName()) {}

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    // TODO: Avoid unnecessary cloning when not downscaling.
    Bitmap scaled_rgb_bitmap = bitmap.CloneAsRGB();

    const int width = bitmap.Width();
    const int height = bitmap.Height();
    const int max_size = std::max(width, height);
    if (max_size > 1000) {
      const double scale =
          static_cast<double>(1000) / static_cast<double>(max_size);
      scaled_rgb_bitmap.Rescale(scale * width, scale * height);
    }

    at::Tensor torch_image = at::empty(
        {1, 3, scaled_rgb_bitmap.Height(), scaled_rgb_bitmap.Width()});
    for (int y = 0; y < scaled_rgb_bitmap.Height(); ++y) {
      for (int x = 0; x < scaled_rgb_bitmap.Width(); ++x) {
        BitmapColor<uint8_t> color;
        CHECK(scaled_rgb_bitmap.GetPixel(x, y, &color));
        constexpr float kNorm = 1.f / 255.f;
        torch_image[0][0][y][x] = kNorm * color.r;
        torch_image[0][1][y][x] = kNorm * color.g;
        torch_image[0][2][y][x] = kNorm * color.b;
      }
    }

    torch::Dict<std::string, torch::Tensor> outputs =
        aliked_.forward(torch_image);

    const auto& torch_keypoints = outputs.at("keypoints");
    const auto& torch_descriptors = outputs.at("descriptors");
    const int num_keypoints = torch_keypoints.size(0);

    keypoints->resize(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      (*keypoints)[i].x =
          0.5f * width * (torch_keypoints[i][0].item<float>() + 1.f);
      (*keypoints)[i].y =
          0.5f * height * (torch_keypoints[i][1].item<float>() + 1.f);
    }

    descriptors->resize(num_keypoints, 128);
    for (int i = 0; i < num_keypoints; ++i) {
      for (int j = 0; j < 128; ++j) {
        (*descriptors)(i, j) = std::min(
            std::max(255.f * torch_descriptors[i][j].item<float>(), 0.f),
            255.f);
      }
    }

    // torch::Dict<std::string, torch::Tensor>
    // ALIKED::run(cv::Mat& img_rgb) {
    //     cv::Mat float_img;
    //     img_rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    //     std::vector<cv::Mat> channels(3);
    //     cv::split(float_img, channels);

    //     auto options = torch::TensorOptions()
    //                        .dtype(torch::kFloat32)
    //                        .device(device_);

    //     std::vector<torch::Tensor> tensor_channels;
    //     tensor_channels.reserve(3);

    //     for (const auto& channel : channels)
    //     {
    //         auto host_tensor = torch::from_blob(
    //             channel.data,
    //             {channel.rows, channel.cols},
    //             torch::TensorOptions().dtype(torch::kFloat32));
    //         tensor_channels.push_back(std::move(host_tensor).to(device_));
    //     }

    //     auto img_tensor = torch::stack(std::move(tensor_channels), 0)
    //                           .unsqueeze(0)
    //                           .to(device_);

    //     // Forward pass with move semantics
    //     auto pred = std::move(*this).forward(std::move(img_tensor));

    //     // Convert keypoints from normalized coordinates to image coordinates
    //     auto kpts = pred.at("keypoints");
    //     const auto h = static_cast<float>(float_img.rows);
    //     const auto w = static_cast<float>(float_img.cols);
    //     const auto wh = torch::tensor({w - 1.0f, h - 1.0f}, kpts.options());
    //     kpts = wh * (kpts + 1) / 2;

    //     pred.insert("keypoints", std::move(kpts));
    //     return pred;
    // }

    // auto torch_keypoints_access = output.accessor<float, 0>();
    // auto torch_descriptors_access = output.accessor<float, 1>();

    // const int num_keypoints = torch_keypoints_access.size(0);

    // descriptors->resize(num_keypoints, 128);
    // for (int i = 0; i < num_keypoints; ++i) {
    //   for (int j = 0; j < 128; ++j) {
    //     (*descriptors)(i, j) = std::min(
    //         std::max(255.f * torch_descriptors_access[i][j], 0.f), 255.f);
    //   }
    // }

    return true;
  }

 private:
  const ALIKEDFeatureOptions options_;
  ALIKED aliked_;
};

}  // namespace

std::unique_ptr<FeatureExtractor> CreateALIKEDFeatureExtractor(
    const ALIKEDFeatureOptions& options) {
  return std::make_unique<ALIKEDFeatureExtractor>(options);
}

}  // namespace colmap
