#pragma once
#include <torch/torch.h>

namespace matcher {
    // Learnable Fourier Positional Encoding
    class LearnableFourierPosEnc : public torch::nn::Module {
    public:
        LearnableFourierPosEnc(int M, int dim, torch::optional<int> F_dim = torch::nullopt, float gamma = 1.0);

        // Forward function returns the position encoding
        torch::Tensor forward(const torch::Tensor& x);

    private:
        float gamma_;
        torch::nn::Linear Wr_{nullptr};
    };
}

