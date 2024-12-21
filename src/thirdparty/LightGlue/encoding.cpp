#include "encoding.hpp"

namespace matcher {
    LearnableFourierPosEnc::LearnableFourierPosEnc(
        int M, int dim, torch::optional<int> F_dim, float gamma)
        : gamma_(gamma) {

        int f_dim = F_dim.value_or(dim);
        // Initialize Wr with normal distribution
        Wr_ = register_module("Wr",
                              torch::nn::Linear(torch::nn::LinearOptions(M, f_dim / 2).bias(false)));

        // Initialize weights according to the paper
        auto std = gamma_ * gamma_;
        torch::nn::init::normal_(Wr_->weight, 0.0, std);
    }

    torch::Tensor LearnableFourierPosEnc::forward(const torch::Tensor& x) {
        // Project and compute trig functions
        auto projected = Wr_->forward(x);
        auto cosines = torch::cos(projected);
        auto sines = torch::sin(projected);

        // Stack and reshape
        auto emb = torch::stack({cosines, sines}, 0).unsqueeze(-3);
        return emb.repeat_interleave(2, -1);
    }
}
