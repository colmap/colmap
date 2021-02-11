// Copyright (c) 2021, Microsoft
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
//
// Author: Antonios Matakos (anmatako-at-microsoft-dot-com)

#ifndef COLMAP_SRC_MVS_TORCH_MODULES_H_
#define COLMAP_SRC_MVS_TORCH_MODULES_H_

#include "torch/torch.h"

namespace colmap {
namespace mvs {

class ConvBnReLU1DImpl : public torch::nn::Module {
 public:
  ConvBnReLU1DImpl(int64_t in_channels, int64_t out_channels,
                   int64_t kernel_size = 3, int64_t stride = 1,
                   int64_t padding = 1, int64_t dilation = 1);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  torch::nn::Conv1d conv{nullptr};
  torch::nn::BatchNorm1d norm{nullptr};
};

TORCH_MODULE(ConvBnReLU1D);

class ConvBnReLU2DImpl : public torch::nn::Module {
 public:
  ConvBnReLU2DImpl(int64_t in_channels, int64_t out_channels,
                   int64_t kernel_size = 3, int64_t stride = 1,
                   int64_t padding = 1, int64_t dilation = 1);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d norm{nullptr};
};

TORCH_MODULE(ConvBnReLU2D);

class ConvBnReLU3DImpl : public torch::nn::Module {
 public:
  ConvBnReLU3DImpl(int64_t in_channels, int64_t out_channels,
                   int64_t kernel_size = 3, int64_t stride = 1,
                   int64_t padding = 1, int64_t dilation = 1);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  torch::nn::Conv3d conv{nullptr};
  torch::nn::BatchNorm3d norm{nullptr};
};

TORCH_MODULE(ConvBnReLU3D);

class RefinementImpl : public torch::nn::Module {
 public:
  RefinementImpl();

  torch::Tensor forward(const torch::Tensor& image,
                        const torch::Tensor& depth_init, const double depth_min,
                        const double depth_max);

 private:
  ConvBnReLU2D conv{nullptr};
  torch::nn::Sequential deconv{nullptr}, residual{nullptr};
};

TORCH_MODULE(Refinement);

class FeatureNetImpl : public torch::nn::Module {
 public:
  FeatureNetImpl();

  std::vector<torch::Tensor> forward(const torch::Tensor& input);

 private:
  torch::nn::Sequential stage1{nullptr}, stage2{nullptr}, stage3{nullptr};
  torch::nn::Conv2d inner1{nullptr}, inner2{nullptr}, output1{nullptr},
      output2{nullptr}, output3{nullptr};
};

TORCH_MODULE(FeatureNet);

class FeatureWeightNetImpl : public torch::nn::Module {
 public:
  FeatureWeightNetImpl(const int num_neighbors, const int num_groups);

  torch::Tensor forward(const torch::Tensor& feature,
                        const torch::Tensor& grid);

 private:
  const int num_neighbors, num_groups;
  torch::nn::Sequential feature_weight{nullptr};
};

TORCH_MODULE(FeatureWeightNet);

class SimilarityNetImpl : public torch::nn::Module {
 public:
  SimilarityNetImpl(const int num_groups);

  torch::Tensor forward(const torch::Tensor& similarity,
                        const torch::Tensor& grid, const torch::Tensor& weight);

 private:
  torch::nn::Sequential conv{nullptr};
};

TORCH_MODULE(SimilarityNet);

class PixelwiseNetImpl : public torch::nn::Module {
 public:
  PixelwiseNetImpl(const int num_groups);

  torch::Tensor forward(const torch::Tensor& input);

 private:
  torch::nn::Sequential conv{nullptr};
};

TORCH_MODULE(PixelwiseNet);

class InitDepthImpl : public torch::nn::Module {
 public:
  InitDepthImpl(const int num_samples, const double interval_scale);

  torch::Tensor forward(const torch::Tensor& depth_init, const double depth_min,
                        const double depth_max, const int64_t batch_size,
                        const int64_t height, const int64_t width,
                        torch::Device device);

 private:
  const int num_samples;
  const double interval_scale;
};

TORCH_MODULE(InitDepth);

class PropagationImpl : public torch::nn::Module {
 public:
  PropagationImpl();

  torch::Tensor forward(const torch::Tensor& depth, const torch::Tensor& grid,
                        const double depth_min, const double depth_max);
};

TORCH_MODULE(Propagation);

class EvaluationImpl : public torch::nn::Module {
 public:
  EvaluationImpl(const int num_groups, const int stage);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& ref_feature_in,
      const torch::TensorList& src_features, const torch::Tensor& ref_proj_mtx,
      const torch::TensorList& src_proj_mtx, const torch::Tensor& depth_init,
      const torch::Tensor& grid, const torch::Tensor& weight,
      const torch::Tensor& view_weights, const bool is_inverse);

 private:
  torch::Tensor DifferentiableWarping(const torch::Tensor& feature,
                                      const torch::Tensor& proj,
                                      const torch::Tensor& ref_proj,
                                      const torch::Tensor& depth);

  torch::Tensor InverseDepthRegression(const torch::Tensor& depth,
                                       const torch::Tensor& score);

  const int num_groups;
  PixelwiseNet pixelwise_net{nullptr};
  SimilarityNet similarity_net{nullptr};
};

TORCH_MODULE(Evaluation);

class PatchMatchModuleImpl : public torch::nn::Module {
 public:
  PatchMatchModuleImpl(const int propagation_range, const int iterations,
                       const int num_samples, const double interval_scale,
                       const int num_features, const int group_correlations,
                       const int propagation_neighbors,
                       const int evaluation_neighbors, const int stage);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& ref_feature, const torch::TensorList& src_features,
      const torch::Tensor& ref_proj_mtx, const torch::TensorList& src_proj_mtx,
      const double depth_min, const double depth_max,
      const torch::Tensor& depth_init, const torch::Tensor& view_weights_init);

 private:
  void CalcOffsets(int dilation);

  torch::Tensor GetGrid(const torch::Tensor& offset,
                        const std::vector<std::array<int, 2>>& orig_offset,
                        const int num_neighbors, const int64_t batch_size,
                        const int64_t height, const int64_t width,
                        torch::Device device);

  torch::Tensor GetDepthWeight(const torch::Tensor& depth,
                               const torch::Tensor& grid,
                               const double depth_min, const double depth_max);

  const int propagation_neighbors, evaluation_neighbors, iterations, stage;
  const double interval_scale;
  std::vector<std::array<int, 2>> prop_offset_orig, eval_offset_orig;
  torch::nn::Conv2d propagation_conv{nullptr}, evaluation_conv{nullptr};
  InitDepth init_depth{nullptr};
  Propagation propagation{nullptr};
  Evaluation evaluation{nullptr};
  FeatureWeightNet feature_weight_net{nullptr};
};

TORCH_MODULE(PatchMatchModule);

class PatchMatchNetModuleImpl : public torch::nn::Module {
 public:
  PatchMatchNetModuleImpl(
      const std::unordered_map<std::string, std::string>& param_dict,
      const std::vector<double>& interval_scale = {0.005, 0.0125, 0.025},
      const std::vector<int>& propagation_range = {6, 4, 2},
      const std::vector<int>& iterations = {1, 2, 2},
      const std::vector<int>& num_samples = {8, 8, 16},
      const std::vector<int>& propagation_neighbors = {0, 8, 16},
      const std::vector<int>& evaluation_neighbors = {9, 9, 9});

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& images, torch::Tensor& proj_matrices,
      const double depth_min, const double depth_max);

  virtual void save(torch::serialize::OutputArchive& archive) const override;
  virtual void load(torch::serialize::InputArchive& archive) override;

 private:
  torch::Tensor CalcConfidence(const torch::Tensor& score);

  const std::unordered_map<std::string, std::string> param_dict;
  const int num_depth;
  FeatureNet feature{nullptr};
  Refinement refinement{nullptr};
  std::vector<PatchMatchModule> patch_match;
};

TORCH_MODULE(PatchMatchNetModule);

inline torch::serialize::OutputArchive& operator<<(
    torch::serialize::OutputArchive& archive, PatchMatchNetModule module) {
  module->save(archive);
  return archive;
}

inline torch::serialize::InputArchive& operator>>(
    torch::serialize::InputArchive& archive, PatchMatchNetModule module) {
  module->load(archive);
  return archive;
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_TORCH_MODULES_H_
