#pragma once
#include <torch/torch.h>

#include <memory>

class SDDH : public torch::nn::Module {
public:
    SDDH(int dims, int kernel_size = 3, int n_pos = 8,
         bool conv2D = false, bool mask = false);

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward(const torch::Tensor& x, const std::vector<torch::Tensor>& keypoints);

private:
    const int kernel_size_;
    const int n_pos_;
    const bool conv2D_;
    const bool mask_;
    torch::nn::Sequential offset_conv_{nullptr};
    torch::nn::Conv2d sf_conv_{nullptr};
    torch::nn::Conv2d convM_{nullptr};
    torch::Tensor agg_weights_;
};
