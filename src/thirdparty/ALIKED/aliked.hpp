#pragma once

#include "blocks.hpp"
#include "input_padder.hpp"

#include <torch/torch.h>

#include <memory>

struct AlikedConfig {
    int c1, c2, c3, c4, dim, K, M;
};

class DKD;
class SDDH;

class ALIKED : public torch::nn::Module {
public:
    explicit ALIKED(const std::string& model_name,
                    const std::string& model_path,
                    const std::string& device = "cuda",
                    int top_k = -1,
                    float scores_th = 0.2,
                    int n_limit = 20000);

    std::tuple<torch::Tensor, torch::Tensor>
    extract_dense_map(const torch::Tensor& image);

    torch::Dict<std::string, torch::Tensor>
    forward(const torch::Tensor& image);

private:
    void init_layers(const std::string& model_name);
    void load_parameters(const std::string& model_path);

    static std::vector<char> get_the_bytes(const std::string& filename);

    torch::nn::AvgPool2d pool2_{nullptr}, pool4_{nullptr};
    std::shared_ptr<ConvBlock> block1_;
    std::shared_ptr<ResBlock> block2_;
    std::shared_ptr<ResBlock> block3_;
    std::shared_ptr<ResBlock> block4_;
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr},
        conv3_{nullptr}, conv4_{nullptr};
    torch::nn::Sequential score_head_{nullptr};

    std::shared_ptr<DKD> dkd_;
    std::shared_ptr<SDDH> desc_head_;

    torch::Device device_;
    int dim_{};
};
