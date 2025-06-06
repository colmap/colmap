#pragma once
#include <torch/torch.h>
#include "core.hpp"


namespace matcher {
    class MatchAssignment : public torch::nn::Module {
    public:
        explicit MatchAssignment(int dim);

        torch::Tensor sigmoid_log_double_softmax(
            const torch::Tensor& sim,
            const torch::Tensor& z0,
            const torch::Tensor& z1);

        torch::Tensor forward(
            const torch::Tensor& desc0,
            const torch::Tensor& desc1);

        torch::Tensor get_matchability(const torch::Tensor& desc);

    private:
        int dim_;
        torch::nn::Linear matchability_{nullptr};
        torch::nn::Linear final_proj_{nullptr};
    };


    class LearnableFourierPosEnc;
    class TokenConfidence;
    class TransformerLayer;
    class MatchAssignment;

    class LightGlue : public torch::nn::Module {
    public:
        explicit LightGlue(int input_dim,
                           const std::string& model_path,
                           const std::string& device = "cuda",
                           const LightGlueConfig& config = LightGlueConfig());

        // Main forward function to process features and find matches
        torch::Dict<std::string, torch::Tensor> forward(
            const torch::Dict<std::string, torch::Tensor>& data0,
            const torch::Dict<std::string, torch::Tensor>& data1);

    private:
        torch::Tensor get_pruning_mask(
            const torch::optional<torch::Tensor>& confidences,
            const torch::Tensor& scores,
            int layer_index);

        bool check_if_stop(
            const torch::Tensor& confidences0,
            const torch::Tensor& confidences1,
            int layer_index,
            int num_points);

    private:
        LightGlueConfig config_;
        torch::Device device_;

        // Neural network components
        torch::nn::Linear input_proj_{nullptr};
        std::shared_ptr<LearnableFourierPosEnc> posenc_;
        std::vector<std::shared_ptr<TransformerLayer>> transformers_;
        std::vector<std::shared_ptr<MatchAssignment>> log_assignment_;
        std::vector<std::shared_ptr<TokenConfidence>> token_confidence_;
        std::vector<float> confidence_thresholds_;

        static const std::unordered_map<std::string, int> pruning_keypoint_thresholds_;
        void load_parameters(const std::string& model_path);
        
        static std::vector<char> get_the_bytes(const std::string& filename);
    };
}
