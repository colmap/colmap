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

const int kNumStages = 4;

PatchMatchNet::PatchMatchNet(const PatchMatchOptions& options,
                             const PatchMatch::Problem& problem)
    : PatchMatch(options, problem),
      dev_in_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      dev_out_(torch::kCPU) {
  InitParamDictionary();
  InitProblemInputs();
}

void PatchMatchNet::Run() {
  // Gradients should be disabled during evaluation to save up on GPU memory
  torch::NoGradGuard no_grad;
  std::cout << "Starting PatchMatchNet" << std::endl;
  torch::Tensor cuda_depth, cuda_confidence;
  PatchMatchNetModule model = PatchMatchNetModule(param_dict_);
  torch::load(model, options_.checkpoint_path);
  model->eval();
  model->to(dev_in_);
  std::tie(cuda_depth, cuda_confidence) = model->forward(
      images_, proj_matrices_, options_.depth_min, options_.depth_max);

  // Copy tensor data to outputs
  torch::Tensor depth = cuda_depth.to(dev_out_);
  const float* depth_ptr = depth.data_ptr<float>();
  std::copy(depth_ptr, depth_ptr + (width_ * height_), depth_map_.GetPtr());

  torch::Tensor confidence = cuda_confidence.to(dev_out_);
  const float* confidence_ptr = confidence.data_ptr<float>();
  std::copy(confidence_ptr, confidence_ptr + (width_ * height_),
            confidence_map_.GetPtr());
}

void PatchMatchNet::InitParamDictionary() {
  if (ExistsFile(options_.param_dict_path)) {
    std::cout << "Reading parameter dictionary from: "
              << options_.param_dict_path << std::endl;
    std::vector<std::string> lines =
        ReadTextFileLines(options_.param_dict_path);
    for (const std::string& line : lines) {
      std::vector<std::string> entry = StringSplit(line, ",");
      if (entry.size() != 2) {
        std::cout << "WARN: skipping malformed param dictionary entry: " << line
                  << std::endl;
        continue;
      }
      StringTrim(&entry[0]);
      StringTrim(&entry[1]);
      param_dict_[entry[0]] = entry[1];
    }
  }
  std::cout << "Loaded dictionary with " << param_dict_.size() << " entries"
            << std::endl;
}

void PatchMatchNet::InitProblemInputs() {
  Image& ref_image = problem_.images->at(problem_.ref_image_idx);
  width_ = ref_image.GetWidth();
  height_ = ref_image.GetHeight();
  depth_map_ =
      DepthMap(width_, height_, options_.depth_min, options_.depth_max);
  confidence_map_ = ConfidenceMap(width_, height_);

  // collect all indexes for ref and src images
  std::vector<int> indexes;
  indexes.push_back(problem_.ref_image_idx);
  indexes.insert(indexes.end(), problem_.src_image_idxs.begin(),
                 problem_.src_image_idxs.end());

  // convert input bitmaps to float tensors of size {1, numImages, 3, height, width}
  // the shape has a leading singleton dimension for the batcn size, and the
  // third dim represents the image RGB channels
  std::vector<torch::Tensor> images(indexes.size());
  for (int i = 0; i < indexes.size(); ++i) {
    std::vector<uint8_t> data =
        problem_.images->at(indexes[i]).GetBitmap().ConvertToRowMajorArray();
    std::vector<float> float_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      float_data[i] = static_cast<float>(data[i]) / 255.0f;
    }
    images[i] = torch::tensor(float_data, torch::device(dev_in_))
                    .view({1, 1, height_, width_, 3})
                    .transpose(1, 4)
                    .squeeze(4);
  }
  images_ = torch::stack(images, 1);
  std::cout << "Created images input with size: " << images_.sizes()
            << std::endl;

  // convert the input projection matrices to float tensors of size {1, kNumStages, numImages, 4, 4}
  // the leading singleton dimension is the batch size
  std::vector<torch::Tensor> stage_matrices(kNumStages);
  float scale = 0.125; // 8x downsampling for the last stage
  for (int stage = kNumStages - 1; stage >= 0; --stage) {
    std::vector<torch::Tensor> proj_matrices(indexes.size());
    for (int i = 0; i < indexes.size(); ++i) {
      // Creating a copy of the image and rescaling to get the correct values in
      // the projection matrix
      Image image(problem_.images->at(indexes[i]));
      image.Rescale(scale);
      // Reformating the projection matrix as a 4x4 flat array
      std::vector<float> data(16, 0.0f);
      std::copy_n(image.GetP(), 12, data.begin());
      data[15] = 1.0f;
      proj_matrices[i] =
          torch::tensor(data, torch::device(dev_in_)).view({1, 4, 4});
    }
    stage_matrices[stage] = torch::stack(proj_matrices, 1);
    // doubling the scale for the next stage
    scale *= 2.0f;
  }
  proj_matrices_ = torch::stack(stage_matrices, 1);
  std::cout << "Created projection matrices input with size: "
            << proj_matrices_.sizes() << std::endl;
}

}  // namespace mvs
}  // namespace colmap
