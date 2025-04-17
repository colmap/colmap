#include "core.hpp"

namespace matcher::utils {
    torch::Tensor normalize_keypoints(
        const torch::Tensor& kpts,
        const torch::optional<torch::Tensor>& size) {

        torch::Tensor size_tensor;
        if (!size.has_value())
        {
            // Compute the size as the range of keypoints
            size_tensor = 1 + std::get<0>(torch::max(kpts, /*dim=*/-2)) - std::get<0>(torch::min(kpts, /*dim=*/-2));
        } else
        {
            // If size is provided but not a tensor, convert it to a tensor
            size_tensor = size.value().to(kpts);
        }

        // Compute shift and scale
        auto shift = size_tensor / 2;
        auto scale = std::get<0>(size_tensor.max(-1)) / 2;

        return (kpts - shift.unsqueeze(-2)) / scale.unsqueeze(-1).unsqueeze(-1);
    }


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    filter_matches(const torch::Tensor& scores, float threshold) {
        int64_t M = scores.size(1) - 1; // 1708
        int64_t N = scores.size(2) - 1; // 1519

        auto scores_slice = scores.slice(1, 0, -1).slice(2, 0, -1);

        // Get max values and indices
        auto max0 = scores_slice.max(2);
        auto max1 = scores_slice.max(1);

        auto m0 = std::get<1>(max0);          // shape: [1, M]
        auto max0_values = std::get<0>(max0); // shape: [1, M]
        auto m1 = std::get<1>(max1);          // shape: [1, N]

        // Create index tensors with correct shape
        auto indices0 = torch::arange(M, m0.options()).unsqueeze(0);
        auto indices1 = torch::arange(N, m1.options()).unsqueeze(0);

        // Ensure all tensors are properly shaped before operations
        m0 = m0.view({1, M});
        m1 = m1.view({1, N});
        indices0 = indices0.view({1, M});
        indices1 = indices1.view({1, N});

        // Calculate mutual matches
        auto mutual0 = indices0 == m1.index_select(1, m0.squeeze()).view({1, M});
        auto mutual1 = indices1 == m0.index_select(1, m1.squeeze()).view({1, N});

        // Calculate scores
        auto max0_exp = max0_values.exp();
        auto zero0 = torch::zeros({1, M}, max0_exp.options());
        auto zero1 = torch::zeros({1, N}, max0_exp.options());
        auto mscores0 = torch::where(mutual0, max0_exp, zero0);

        // Ensure proper shapes for score calculation
        auto mscores0_expanded = mscores0.index_select(1, m1.squeeze());
        auto mscores1 = torch::where(mutual1, mscores0_expanded.view({1, N}), zero1);

        // Calculate valid matches
        auto valid0 = mutual0 & (mscores0 > threshold);
        auto valid1 = mutual1 & (mscores1 > threshold);

        // Create output tensors with correct shape
        auto m0_valid = torch::where(valid0, m0, torch::full({1, M}, -1, m0.options()));
        auto m1_valid = torch::where(valid1, m1, torch::full({1, N}, -1, m1.options()));

        return std::make_tuple(m0_valid, m1_valid, mscores0, mscores1);
    }
}