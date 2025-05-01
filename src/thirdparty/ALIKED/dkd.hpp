#pragma once
#include <torch/torch.h>

#include <array>

class DKD : public torch::nn::Module {
public:
    DKD(int radius = 2, int top_k = -1, float scores_th = 0.2, int n_limit = 20000);

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    detect_keypoints(const torch::Tensor& scores_map, bool sub_pixel = true);

    torch::Tensor simple_nms(const torch::Tensor& scores, int nms_radius);

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward(const torch::Tensor& scores_map, bool sub_pixel = true);

private:
    static constexpr int calculateKernelSize(int radius) { return 2 * radius + 1; }

    const int radius_;
    const int top_k_;
    const float scores_th_;
    const int n_limit_;
    const int kernel_size_;
    const float temperature_;
    torch::nn::Unfold unfold_{nullptr};
    torch::Tensor hw_grid_;
};
