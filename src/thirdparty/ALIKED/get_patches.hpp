#pragma once

#include <torch/torch.h>

namespace custom_ops {

    torch::Tensor get_patches_forward(const torch::Tensor& map, torch::Tensor& points, int64_t radius);
    torch::Tensor get_patches_backward(const torch::Tensor& d_patches, torch::Tensor& points, int64_t H, int64_t W);
} // namespace custom_ops
