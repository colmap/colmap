#pragma once

#include <torch/torch.h>
#include <memory>

namespace matcher{
    class SelfBlock;
    class CrossBlock;

    class TransformerLayer : public torch::nn::Module {
    public:
        TransformerLayer(int embed_dim, int num_heads, bool flash = false, bool bias = true);

        std::tuple<torch::Tensor, torch::Tensor> forward(
            const torch::Tensor& desc0,
            const torch::Tensor& desc1,
            const torch::Tensor& encoding0,
            const torch::Tensor& encoding1);

    private:
        std::shared_ptr<SelfBlock> self_attn_;
        std::shared_ptr<CrossBlock> cross_attn_;
    };
}
