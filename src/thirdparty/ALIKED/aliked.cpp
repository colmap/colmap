#include "aliked.hpp"

#include "dkd.hpp"
#include "sddh.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <unordered_map>

namespace fs = std::filesystem;

// Static configuration map
inline const std::unordered_map<std::string, AlikedConfig> ALIKED_CFGS = {
    {"aliked-t16", {8, 16, 32, 64, 64, 3, 16}},
    {"aliked-n16", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n16rot", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n32", {16, 32, 64, 128, 128, 3, 32}}};

ALIKED::ALIKED(const std::string& model_name,
               const std::string& model_path,
               const std::string& device,
               int top_k,
               float scores_th,
               int n_limit)
    : device_(device),
      dim_(-1) {

    // Initialize DKD and descriptor head
    dkd_ = std::make_shared<DKD>(2, top_k, scores_th, n_limit);
    const auto& config = ALIKED_CFGS.at(std::string(model_name));
    desc_head_ = std::make_shared<SDDH>(config.dim, config.K, config.M);

    // Initialize layers
    init_layers(model_name);

    // Load weights first
    load_parameters(model_path);

    // Move everything to the specified device
    this->to(device_);
    dkd_->to(device_);       // Explicitly move DKD
    desc_head_->to(device_); // Explicitly move SDDH

    // Double check all submodules are on the correct device
    for (const auto& param : this->parameters())
    {
        if (param.device() != device_)
        {
            param.to(device_);
        }
    }

    for (const auto& buffer : this->buffers())
    {
        if (buffer.device() != device_)
        {
            buffer.to(device_);
        }
    }

    this->eval();
}

std::tuple<torch::Tensor, torch::Tensor>
ALIKED::extract_dense_map(torch::Tensor image) && {
    // Create padder for input
    auto padder = InputPadder(image.size(2), image.size(3), 32);
    image = std::move(padder).pad(std::move(image));

    // Feature extraction with move semantics
    auto x1 = std::dynamic_pointer_cast<ConvBlock>(block1_)->forward(image);
    auto x2 = std::dynamic_pointer_cast<ResBlock>(block2_)->forward(pool2_->forward(x1));
    auto x3 = std::dynamic_pointer_cast<ResBlock>(block3_)->forward(pool4_->forward(x2));
    auto x4 = std::dynamic_pointer_cast<ResBlock>(block4_)->forward(pool4_->forward(x3));

    // Feature aggregation
    auto x1_processed = torch::selu(conv1_->forward(x1));
    auto x2_processed = torch::selu(conv2_->forward(x2));
    auto x3_processed = torch::selu(conv3_->forward(x3));
    auto x4_processed = torch::selu(conv4_->forward(x4));

    // Upsample with move semantics
    auto options = torch::nn::functional::InterpolateFuncOptions()
                       .mode(torch::kBilinear)
                       .align_corners(true);

    auto x2_up = torch::nn::functional::interpolate(x2_processed,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));
    auto x3_up = torch::nn::functional::interpolate(x3_processed,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));
    auto x4_up = torch::nn::functional::interpolate(x4_processed,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));

    auto x1234 = torch::cat({std::move(x1_processed),
                             std::move(x2_up),
                             std::move(x3_up),
                             std::move(x4_up)},
                            1);

    // Generate score map and feature map
    auto score_map = torch::sigmoid(score_head_->forward(x1234.clone()));
    auto feature_map = torch::nn::functional::normalize(x1234,
                                                        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // Unpad tensors with move semantics
    feature_map = std::move(padder).unpad(std::move(feature_map));
    score_map = std::move(padder).unpad(std::move(score_map));

    return std::make_tuple(std::move(feature_map), std::move(score_map));
}

torch::Dict<std::string, torch::Tensor>
ALIKED::forward(torch::Tensor image) && {
    // Tensors to the right device
    image = image.to(device_);

    auto start_time = std::chrono::high_resolution_clock::now();

    auto [feature_map, score_map] = std::move(*this).extract_dense_map(std::move(image));
    auto [keypoints, kptscores, scoredispersitys] = std::move(*dkd_).forward(score_map);
    auto [descriptors, offsets] = std::move(*desc_head_).forward(feature_map, keypoints);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0f;

    torch::Dict<std::string, torch::Tensor> output;
    output.insert("keypoints", std::move(keypoints[0]));
    output.insert("descriptors", std::move(descriptors[0]));
    output.insert("scores", std::move(kptscores[0]));
    output.insert("score_dispersity", std::move(scoredispersitys[0]));
    output.insert("score_map", std::move(score_map));
    output.insert("time", torch::tensor(duration));

    return output;
}

torch::Dict<std::string, torch::Tensor>
ALIKED::forward(const torch::Tensor& image) & {
    auto image_copy = image.clone();
    return std::move(*this).forward(std::move(image_copy));
}

void ALIKED::init_layers(const std::string& model_name) {
    const auto& config = ALIKED_CFGS.at(std::string(model_name));
    dim_ = config.dim;

    // Basic layers
    pool2_ = register_module("pool2",
                             torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
    pool4_ = register_module("pool4",
                             torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)));

    // Blocks with move semantics
    block1_ = register_module(
        "block1",
        std::make_shared<ConvBlock>(3, config.c1, "conv", false));

    auto downsample2 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.c1, config.c2, 1));
    block2_ = register_module(
        "block2",
        std::make_shared<ResBlock>(config.c1, config.c2, 1, downsample2, "conv"));

    auto downsample3 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.c2, config.c3, 1));
    block3_ = register_module(
        "block3",
        std::make_shared<ResBlock>(config.c2, config.c3, 1, downsample3, "dcn"));

    auto downsample4 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(config.c3, config.c4, 1));
    block4_ = register_module(
        "block4",
        std::make_shared<ResBlock>(config.c3, config.c4, 1, downsample4, "dcn"));

    // Convolution layers
    const int out_channels = dim_ / 4;
    conv1_ = register_module("conv1",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c1, out_channels, 1).stride(1).bias(false)));
    conv2_ = register_module("conv2",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c2, out_channels, 1).stride(1).bias(false)));
    conv3_ = register_module("conv3",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c3, out_channels, 1).stride(1).bias(false)));
    conv4_ = register_module("conv4",
                             torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c4, out_channels, 1).stride(1).bias(false)));

    // Score head
    torch::nn::Sequential score_head;
    score_head->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(dim_, 8, 1).stride(1).bias(false)));
    score_head->push_back(torch::nn::SELU());
    score_head->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(8, 4, 3).padding(1).stride(1).bias(false)));
    score_head->push_back(torch::nn::SELU());
    score_head->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(4, 4, 3).padding(1).stride(1).bias(false)));
    score_head->push_back(torch::nn::SELU());
    score_head->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(4, 1, 3).padding(1).stride(1).bias(false)));

    score_head_ = register_module("score_head", score_head);
    register_module("desc_head", desc_head_);
    register_module("dkd", dkd_);
}

void ALIKED::load_parameters(const std::string& model_path) {
    auto f = get_the_bytes(model_path);
    auto weights = torch::pickle_load(f).toGenericDict();

    // Use unordered_maps for O(1) lookup
    std::unordered_map<std::string, torch::Tensor> param_map;
    std::unordered_map<std::string, torch::Tensor> buffer_map;

    auto model_params = named_parameters();
    auto model_buffers = named_buffers();
    // Pre-allocate with expected size
    param_map.reserve(model_params.size());
    buffer_map.reserve(model_buffers.size());

    // Collect parameter names
    for (const auto& p : model_params)
    {
        param_map.emplace(p.key(), p.value());
    }

    // Collect buffer names
    for (const auto& b : model_buffers)
    {
        buffer_map.emplace(b.key(), b.value());
    }

    // Update parameters and buffers
    torch::NoGradGuard no_grad;

    for (const auto& w : weights)
    {
        const auto name = w.key().toStringRef();
        const auto& param = w.value().toTensor();

        // Try parameters first
        if (auto it = param_map.find(name); it != param_map.end())
        {
            if (it->second.sizes() == param.sizes())
            {
                it->second.copy_(param);
            } else
            {
                throw std::runtime_error(
                    "Shape mismatch for parameter: " + name +
                    " Expected: " + std::to_string(it->second.numel()) +
                    " Got: " + std::to_string(param.numel()));
            }
            continue;
        }

        // Then try buffers
        if (auto it = buffer_map.find(name); it != buffer_map.end())
        {
            if (it->second.sizes() == param.sizes())
            {
                it->second.copy_(param);
            } else
            {
                throw std::runtime_error(
                    "Shape mismatch for buffer: " + name +
                    " Expected: " + std::to_string(it->second.numel()) +
                    " Got: " + std::to_string(param.numel()));
            }
            continue;
        }

        // Parameter not found in model
        std::cerr << "Warning: " << name
                  << " not found in model parameters or buffers\n";
    }
}

std::vector<char> ALIKED::get_the_bytes(const std::string& filename) {
    // Use RAII file handling
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Pre-allocate vector
    std::vector<char> buffer;
    buffer.reserve(size);

    // Read file in chunks for better performance
    constexpr size_t CHUNK_SIZE = 8192;
    char chunk[CHUNK_SIZE];

    while (file.read(chunk, CHUNK_SIZE))
    {
        buffer.insert(buffer.end(), chunk, chunk + file.gcount());
    }
    if (file.gcount() > 0)
    {
        buffer.insert(buffer.end(), chunk, chunk + file.gcount());
    }

    return buffer;
}
