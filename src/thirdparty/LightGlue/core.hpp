#pragma once
#include <string>
#include <torch/torch.h>

namespace matcher {
    struct LightGlueConfig {
        static constexpr float DEFAULT_DEPTH_CONFIDENCE = 0.95f;
        static constexpr float DEFAULT_WIDTH_CONFIDENCE = 0.99f;
        static constexpr float DEFAULT_FILTER_THRESHOLD = 0.1f;

        std::string name{"lightglue"};
        int input_dim{128};
        int descriptor_dim{256};
        bool add_scale_ori{false};
        int n_layers{9};
        int num_heads{4};
        bool flash{true};
        bool mp{false};
        float depth_confidence{DEFAULT_DEPTH_CONFIDENCE};
        float width_confidence{DEFAULT_WIDTH_CONFIDENCE};
        float filter_threshold{DEFAULT_FILTER_THRESHOLD};
    };
}

namespace matcher::utils {
    torch::Tensor normalize_keypoints(const torch::Tensor& kpts,
                                             const torch::optional<torch::Tensor>& size = torch::nullopt);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    filter_matches(const torch::Tensor& scores, float threshold);
}