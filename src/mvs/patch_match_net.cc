// Copyright (c) 2021, Microsoft.
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

#include "mvs/patch_match_net.h"
#include "util/misc.h"
#include "torch/script.h"

namespace colmap {
namespace mvs {

static const torch::Device kDevOut(torch::kCPU);
static const torch::Device kDevIn(torch::cuda::is_available() ? torch::kCUDA
                                                              : torch::kCPU);

std::unordered_map<int, torch::jit::Module> PatchMatchNet::model_;

PatchMatchNet::PatchMatchNet(const PatchMatchOptions& options,
                             const PatchMatch::Problem& problem,
                             const int thread_index)
    : PatchMatch(options, problem), thread_index_(thread_index) {
  InitModule();
  InitProblemInputs();
}

void PatchMatchNet::Run() {
  // Gradients should be disabled during evaluation to save up on GPU memory
  torch::NoGradGuard no_grad;
  std::cout << "Starting PatchMatchNet..." << std::endl;

  // Run TorchScript module
  torch::IValue result = model_[thread_index_].forward(
      {images_, intrinsics_, extrinsics_, depth_params_});

  // Copy tensor data to outputs
  size_t width = problem_.images->at(problem_.ref_image_idx).GetWidth();
  size_t height = problem_.images->at(problem_.ref_image_idx).GetHeight();
  torch::Tensor depth = result.toTuple()->elements()[0].toTensor().to(kDevOut);
  const float* depth_ptr = depth.data_ptr<float>();
  std::copy(depth_ptr, depth_ptr + (width * height), depth_map_.GetPtr());

  torch::Tensor confidence =
      result.toTuple()->elements()[1].toTensor().to(kDevOut);
  const float* confidence_ptr = confidence.data_ptr<float>();
  std::copy(confidence_ptr, confidence_ptr + (width * height),
            confidence_map_.GetPtr());
}

void PatchMatchNet::InitModule() {
  torch::jit::setGraphExecutorOptimize(true);
  // Load module only once per thread index from the execution thread-pool.
  // Ensures we use execution optimization in a thread-safe manner
  if (model_.count(thread_index_) == 0) {
    std::cout << "First definition of patch-match module for thread index: "
              << options_.gpu_index << std::endl;
    model_[thread_index_] =
        torch::jit::load(options_.mvs_module_path, kDevIn);
  } else {
    std::cout << "Patch-match module already defined for thread index: "
              << options_.gpu_index << std::endl;
  }

  std::cout << "PatchMatchNet: Device input type: " << kDevIn << std::endl;
}

void PatchMatchNet::InitProblemInputs() {
  Image& ref_image = problem_.images->at(problem_.ref_image_idx);
  size_t ref_width = ref_image.GetWidth();
  size_t ref_height = ref_image.GetHeight();
  depth_map_ =
      DepthMap(ref_width, ref_height, options_.depth_min, options_.depth_max);
  confidence_map_ = ConfidenceMap(ref_width, ref_height);

  // Create tensor from depth_min and depth_max
  std::vector<float> depth_data(
      {(float)options_.depth_min, (float)options_.depth_max});
  depth_params_ = torch::tensor(depth_data, torch::device(kDevIn)).view({1, 2});

  // collect all indexes for ref and src images
  std::vector<int> indexes;
  indexes.push_back(problem_.ref_image_idx);
  indexes.insert(indexes.end(), problem_.src_image_idxs.begin(),
                 problem_.src_image_idxs.end());

  // convert input bitmaps, intrinsics, and extrinsics to float tensors of size
  // {1, numImages, 3, height, width}, {1, numImages, 3, 3}, and {1, numImages,
  // 4, 4} respectively. The shape has a leading singleton dimension for the
  // batcn size, and the third dim of the images tensor represents the image RGB
  // channels
  images_.resize(indexes.size());
  std::vector<torch::Tensor> intrinsics(indexes.size());
  std::vector<torch::Tensor> extrinsics(indexes.size());
  for (int i = 0; i < indexes.size(); ++i) {
    const Image& image = problem_.images->at(indexes[i]);

    // Read bitmap uint data and convert to float in range [0, 1]
    std::vector<uint8_t> bitmap_data =
        image.GetBitmap().ConvertToRowMajorArray();
    std::vector<float> image_data(bitmap_data.size());
    for (size_t i = 0; i < bitmap_data.size(); ++i) {
      image_data[i] = static_cast<float>(bitmap_data[i]) / 255.0f;
    }
    images_[i] = torch::tensor(image_data, torch::device(kDevIn))
                     .view({1, 1, (int64_t)image.GetHeight(),
                            (int64_t)image.GetWidth(), 3})
                     .transpose(1, 4)
                     .squeeze(4);

    // Copying the intrinsics matrix data in 3x3 tensor
    std::vector<float> intrinsic_data(image.GetK(), image.GetK() + 9);
    intrinsics[i] =
        torch::tensor(intrinsic_data, torch::device(kDevIn)).view({1, 3, 3});

    // Reformating the intrinsics matrix as a 4x4 flat array
    std::vector<float> extrinsic_data(16, 0.0f);
    ComposePoseMatrix(image.GetR(), image.GetT(), extrinsic_data.data());
    extrinsic_data[15] = 1.0f;
    extrinsics[i] =
        torch::tensor(extrinsic_data, torch::device(kDevIn)).view({1, 4, 4});
  }
  intrinsics_ = torch::stack(intrinsics, 1);
  extrinsics_ = torch::stack(extrinsics, 1);
  std::cout << "Created image, intrinsinc, and extrinsic inputs with size: "
            << images_[0].sizes() << ", " << intrinsics_.sizes() << ", "
            << extrinsics_.sizes() << std::endl;
}

}  // namespace mvs
}  // namespace colmap
