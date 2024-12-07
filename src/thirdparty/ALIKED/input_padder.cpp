#include "input_padder.hpp"

torch::Tensor InputPadder::pad(torch::Tensor x) && {
    return torch::nn::functional::pad(
        std::move(x),
        torch::nn::functional::PadFuncOptions({pad_[0], pad_[1], pad_[2], pad_[3]})
            .mode(torch::kReplicate));
}

torch::Tensor InputPadder::pad(const torch::Tensor& x) & {
    return torch::nn::functional::pad(
        x,
        torch::nn::functional::PadFuncOptions({pad_[0], pad_[1], pad_[2], pad_[3]})
            .mode(torch::kReplicate));
}

[[maybe_unused]] torch::Tensor InputPadder::unpad(torch::Tensor x) && {
    int h = x.size(-2);
    int w = x.size(-1);
    return std::move(x).index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(pad_[2], h - pad_[3]),
                               torch::indexing::Slice(pad_[0], w - pad_[1])});
}

torch::Tensor InputPadder::unpad(const torch::Tensor& x) & {
    int h = x.size(-2);
    int w = x.size(-1);
    return x.index({torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(pad_[2], h - pad_[3]),
                    torch::indexing::Slice(pad_[0], w - pad_[1])});
}

void InputPadder::setPadding(const std::array<int, 4>& pad) {
    pad_ = pad;
}