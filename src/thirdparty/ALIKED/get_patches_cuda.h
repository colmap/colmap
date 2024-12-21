#pragma once
#include <torch/torch.h>
// CUDA declarations
torch::Tensor get_patches_forward_cuda(const torch::Tensor& map, torch::Tensor& points, int64_t radius);
torch::Tensor get_patches_backward_cuda(const torch::Tensor& d_patches, torch::Tensor& points, int64_t H, int64_t W);
