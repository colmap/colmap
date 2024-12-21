#include "transformer.hpp"
#include "attention.hpp"

namespace matcher {
    TransformerLayer::TransformerLayer(int embed_dim, int num_heads, bool flash, bool bias) {
        // Initialize self-attention block
        self_attn_ = register_module("self_attn",
                                     std::make_shared<SelfBlock>(embed_dim, num_heads, flash, bias));

        // Initialize cross-attention block
        cross_attn_ = register_module("cross_attn",
                                      std::make_shared<CrossBlock>(embed_dim, num_heads, flash, bias));
    }

    std::tuple<torch::Tensor, torch::Tensor> TransformerLayer::forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1,
        const torch::Tensor& encoding0,
        const torch::Tensor& encoding1) {

        auto desc0_sa = self_attn_->forward(desc0, encoding0);
        auto desc1_sa = self_attn_->forward(desc1, encoding1);

        // Apply cross-attention between the two sets
        return cross_attn_->forward(desc0_sa, desc1_sa);
    }
}