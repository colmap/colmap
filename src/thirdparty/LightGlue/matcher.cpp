#include "matcher.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include "attention.hpp"
#include "encoding.hpp"
#include "transformer.hpp"
#include <torch/torch.h>

namespace {
std::string map_python_to_cpp(const std::string& python_name) {
  std::string cpp_name = python_name;

  size_t pos_transformer = cpp_name.find("transformers");
  size_t pos_assignment = cpp_name.find("log_assignment");
  size_t pos_confidence = cpp_name.find("token_confidence");

  size_t pos = std::min({pos_transformer, pos_assignment, pos_confidence});
  // Replace ".<digit>" with "<digit>"
  size_t dot_pos = cpp_name.find_first_of("0123456789", pos);
  if (dot_pos != std::string::npos && cpp_name[dot_pos - 1] == '.') {
    cpp_name.erase(dot_pos - 1, 1);  // Remove the dot before the number
  }

  return cpp_name;
}
}  // namespace

namespace matcher {
MatchAssignment::MatchAssignment(int dim)
    : dim_(dim),
      matchability_(torch::nn::Linear(torch::nn::LinearOptions(dim_, 1).bias(
          true))),  // Adjust the dimensions as needed
      final_proj_(torch::nn::LinearOptions(dim_, dim_)
                      .bias(true))  // Adjust the dimensions as needed
{
  register_module("matchability", matchability_);
  register_module("final_proj", final_proj_);
}

torch::Tensor MatchAssignment::forward(const torch::Tensor& desc0,
                                       const torch::Tensor& desc1) {
  // Project descriptors
  auto mdesc0 = final_proj_->forward(desc0);
  auto mdesc1 = final_proj_->forward(desc1);

  // Scale by dimension
  auto d = mdesc0.size(-1);
  auto scale = 1.0f / std::pow(d, 0.25f);
  mdesc0 = mdesc0 * scale;
  mdesc1 = mdesc1 * scale;
  auto sim = torch::einsum("bmd,bnd->bmn", {mdesc0, mdesc1});
  auto z0 = matchability_->forward(desc0);
  auto z1 = matchability_->forward(desc1);
  auto scores = sigmoid_log_double_softmax(sim, z0, z1);
  return scores;
}

torch::Tensor MatchAssignment::get_matchability(const torch::Tensor& desc) {
  // Debug input tensor
  auto weight = matchability_->weight.data();
  auto bias = matchability_->bias.data();
  auto result = torch::sigmoid(matchability_->forward(desc)).squeeze(-1);
  return result;
}

torch::Tensor MatchAssignment::sigmoid_log_double_softmax(
    const torch::Tensor& sim,
    const torch::Tensor& z0,
    const torch::Tensor& z1) {
  auto batch_size = sim.size(0);
  auto m = sim.size(1);
  auto n = sim.size(2);

  auto certainties =
      torch::log_sigmoid(z0) + torch::log_sigmoid(z1).transpose(1, 2);
  auto scores0 = torch::log_softmax(sim, 2);
  auto scores1 = torch::log_softmax(sim.transpose(-1, -2).contiguous(), 2)
                     .transpose(-1, -2);

  auto scores = torch::full(
      {batch_size, m + 1, n + 1},
      0.0f,
      torch::TensorOptions().device(sim.device()).dtype(sim.dtype()));

  scores.index_put_({torch::indexing::Slice(),
                     torch::indexing::Slice(torch::indexing::None, m),
                     torch::indexing::Slice(torch::indexing::None, n)},
                    scores0 + scores1 + certainties);

  scores.index_put_({torch::indexing::Slice(),
                     torch::indexing::Slice(torch::indexing::None, -1),
                     n},
                    torch::log_sigmoid(-z0.squeeze(-1)));

  scores.index_put_({torch::indexing::Slice(),
                     m,
                     torch::indexing::Slice(torch::indexing::None, -1)},
                    torch::log_sigmoid(-z1.squeeze(-1)));

  return scores;
}
// Static member initialization
const std::unordered_map<std::string, int>
    LightGlue::pruning_keypoint_thresholds_ = {
        {"cpu", -1}, {"mps", -1}, {"cuda", 1024}, {"flash", 1536}};

LightGlue::LightGlue(int input_dim,
                     const std::string& model_path,
                     const std::string& device,
                     const LightGlueConfig& config)
    : config_(config), device_(device) {

  if (config_.flash) {
    at::globalContext().setSDPUseFlash(true);
  }

  config_.input_dim = input_dim;

  // Initialize input projection if needed
  if (config_.input_dim != config_.descriptor_dim) {
    input_proj_ = register_module(
        "input_proj",
        torch::nn::Linear(config_.input_dim, config_.descriptor_dim));
  }

  // Initialize positional encoding
  posenc_ = register_module("posenc",
                            std::make_shared<LearnableFourierPosEnc>(
                                2 + 2 * config_.add_scale_ori,
                                config_.descriptor_dim / config_.num_heads,
                                config_.descriptor_dim / config_.num_heads));

  // Initialize transformer layers
  for (int i = 0; i < config_.n_layers; ++i) {
    auto layer = std::make_shared<TransformerLayer>(
        config_.descriptor_dim, config_.num_heads, config_.flash);

    transformers_.push_back(layer);
    register_module("transformers" + std::to_string(i), layer);
  }

  // Initialize assignment and token confidence layers
  for (int i = 0; i < config_.n_layers; ++i) {
    auto assign = std::make_shared<MatchAssignment>(config_.descriptor_dim);
    log_assignment_.push_back(assign);
    register_module("log_assignment" + std::to_string(i), assign);

    if (i < config_.n_layers - 1) {
      auto conf = std::make_shared<TokenConfidence>(config_.descriptor_dim);
      token_confidence_.push_back(conf);
      register_module("token_confidence" + std::to_string(i), conf);
    }
  }

  // Register confidence thresholds buffer
  confidence_thresholds_.reserve(config_.n_layers);

  auto confidence_threshold = [](int layer_index, int n_layers) -> float {
    float progress = static_cast<float>(layer_index) / n_layers;
    float threshold = 0.8f + 0.1f * std::exp(-4.0f * progress);
    return std::clamp(threshold, 0.0f, 1.0f);
  };

  for (int i = 0; i < config_.n_layers; ++i) {
    confidence_thresholds_.push_back(confidence_threshold(i, config.n_layers));
  }

  // Load weights if specified
  load_parameters(model_path);
  torch::nn::Module::to(device_);
}

torch::Tensor LightGlue::get_pruning_mask(
    const torch::optional<torch::Tensor>& confidences,
    const torch::Tensor& scores,
    int layer_index) {
  // Initialize keep mask based on scores
  auto keep = scores > (1.0f - config_.width_confidence);

  // Include low-confidence points if confidences are provided
  if (confidences.has_value()) {
    keep = keep | (confidences.value() <= confidence_thresholds_[layer_index]);
  }

  return keep;
}

bool LightGlue::check_if_stop(const torch::Tensor& confidences0,
                              const torch::Tensor& confidences1,
                              int layer_index,
                              int num_points) {
  auto confidences = torch::cat({confidences0, confidences1}, -1);
  auto threshold = confidence_thresholds_[layer_index];
  auto ratio_confident =
      1.0f - (confidences < threshold).to(torch::kFloat32).sum().item<float>() /
                 num_points;
  return ratio_confident > config_.depth_confidence;
}

torch::Dict<std::string, torch::Tensor> LightGlue::forward(
    const torch::Dict<std::string, torch::Tensor>& data0,
    const torch::Dict<std::string, torch::Tensor>& data1) {
  // Extract keypoints and descriptors
  // TODO: Batching
  const auto& kpts0_ref = data0.at("keypoints");
  const auto& kpts1_ref = data1.at("keypoints");
  const auto& desc0_ref = data0.at("descriptors");
  const auto& desc1_ref = data1.at("descriptors");

  // Single operation instead of multiple calls
  auto kpts0 = kpts0_ref.detach().contiguous().unsqueeze(0);
  auto kpts1 = kpts1_ref.detach().contiguous().unsqueeze(0);
  auto desc0 = desc0_ref.detach().contiguous().unsqueeze(0);
  auto desc1 = desc1_ref.detach().contiguous().unsqueeze(0);

  // Move all to the same device as the model
  kpts0 = kpts0.to(device_);
  kpts1 = kpts1.to(device_);
  desc0 = desc0.to(device_);
  desc1 = desc1.to(device_);

  // Pre-calculate sizes once
  const int64_t b = kpts0.size(0);
  const int64_t m = kpts0.size(1);
  const int64_t n = kpts1.size(1);

  // Get image sizes if available
  torch::optional<torch::Tensor> size0, size1;
  if (data0.contains("image_size")) size0 = data0.at("image_size");
  if (data1.contains("image_size")) size1 = data1.at("image_size");

  // Normalize keypoints
  kpts0 = matcher::utils::normalize_keypoints(kpts0, size0).clone();
  kpts1 = matcher::utils::normalize_keypoints(kpts1, size1).clone();

  // Add scale and orientation if configured
  if (config_.add_scale_ori) {
    kpts0 = torch::cat({kpts0,
                        data0.at("scales").unsqueeze(0),
                        data0.at("oris").unsqueeze(0)},
                       -1);
    kpts1 = torch::cat({kpts1,
                        data1.at("scales").unsqueeze(0),
                        data1.at("oris").unsqueeze(0)},
                       -1);
  }

  // Convert to fp16 if mixed precision is enabled
  if (config_.mp && device_.is_cuda()) {
    desc0 = desc0.to(torch::kHalf);
    desc1 = desc1.to(torch::kHalf);
  }

  // Project descriptors if needed
  if (config_.input_dim != config_.descriptor_dim) {
    desc0 = input_proj_->forward(desc0);
    desc1 = input_proj_->forward(desc1);
  }

  // Generate positional encodings
  auto encoding0 = posenc_->forward(kpts0);
  auto encoding1 = posenc_->forward(kpts1);

  // Initialize pruning if enabled
  const bool do_early_stop = config_.depth_confidence > 0.f;
  const bool do_point_pruning = config_.width_confidence > 0.f;
  const auto pruning_th =
      pruning_keypoint_thresholds_.at(config_.flash       ? "flash"
                                      : device_.is_cuda() ? "cuda"
                                                          : "cpu");

  torch::Tensor ind0, ind1, prune0, prune1;
  if (do_point_pruning) {
    ind0 =
        torch::arange(m, torch::TensorOptions().device(device_)).unsqueeze(0);
    ind1 =
        torch::arange(n, torch::TensorOptions().device(device_)).unsqueeze(0);
    prune0 = torch::ones_like(ind0);
    prune1 = torch::ones_like(ind1);
  }

  // Process through transformer layers
  torch::optional<torch::Tensor> token0, token1;
  int last_layer = 0;
  for (int i = 0; i < config_.n_layers; ++i) {
    last_layer = i;
    if (desc0.size(1) == 0 || desc1.size(1) == 0) break;

    std::tie(desc0, desc1) =
        transformers_[i]->forward(desc0, desc1, encoding0, encoding1);

    if (i == config_.n_layers - 1) continue;

    // Early stopping check
    if (do_early_stop) {
      std::tie(token0, token1) = token_confidence_[i]->forward(desc0, desc1);

      if (check_if_stop(token0.value().index(
                            {torch::indexing::Slice(),
                             torch::indexing::Slice(torch::indexing::None, m)}),
                        token1.value().index(
                            {torch::indexing::Slice(),
                             torch::indexing::Slice(torch::indexing::None, n)}),
                        i,
                        m + n)) {
        break;
      }
    }

    if (do_point_pruning && desc0.size(-2) > pruning_th) {
      auto scores0 = log_assignment_[i]->get_matchability(desc0);
      auto prunemask0 = get_pruning_mask(token0, scores0, i);

      if (prunemask0.dtype() != torch::kBool) {
        prunemask0 = prunemask0.to(torch::kBool);
      }

      auto where_result = torch::where(prunemask0);
      auto keep0 = where_result[1];

      if (keep0.numel() > 0) {
        ind0 = ind0.index_select(1, keep0);
        desc0 = desc0.index_select(1, keep0);
        encoding0 = encoding0.index_select(-2, keep0);
        prune0.index_put_({torch::indexing::Slice(), ind0},
                          prune0.index({torch::indexing::Slice(), ind0}) + 1);
      } else {
        std::cout << "No points kept after pruning for desc0." << std::endl;
      }
    }

    if (do_point_pruning && desc1.size(-2) > pruning_th) {
      auto scores1 = log_assignment_[i]->get_matchability(desc1);
      auto prunemask1 = get_pruning_mask(token1, scores1, i);
      if (prunemask1.dtype() != torch::kBool) {
        prunemask1 = prunemask1.to(torch::kBool);
      }

      auto where_result = torch::where(prunemask1);
      auto keep1 = where_result[1];
      if (keep1.numel() > 0) {
        ind1 = ind1.index_select(1, keep1);
        desc1 = desc1.index_select(1, keep1);
        encoding1 = encoding1.index_select(-2, keep1);
        prune1.index_put_({torch::indexing::Slice(), ind1},
                          prune1.index({torch::indexing::Slice(), ind1}) + 1);
      } else {
        std::cout << "No points kept after pruning for desc1." << std::endl;
      }
    }
  }

  // Handle empty descriptor case
  if (desc0.size(1) == 0 || desc1.size(1) == 0) {
    auto m0 = torch::full(
        {b, m}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto m1 = torch::full(
        {b, n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto mscores0 = torch::zeros({b, m}, device_);
    auto mscores1 = torch::zeros({b, n}, device_);

    if (!do_point_pruning) {
      prune0 = torch::ones_like(mscores0) * config_.n_layers;
      prune1 = torch::ones_like(mscores1) * config_.n_layers;
    }

    torch::Dict<std::string, torch::Tensor> output;
    output.insert("matches0", m0);
    output.insert("matches1", m1);
    output.insert("matching_scores0", mscores0);
    output.insert("matching_scores1", mscores1);
    output.insert("stop", torch::tensor(last_layer + 1));
    output.insert("prune0", prune0);
    output.insert("prune1", prune1);

    return output;
  }

  // Remove padding and compute assignment
  desc0 = desc0.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, m),
                       torch::indexing::Slice()});
  desc1 = desc1.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, n),
                       torch::indexing::Slice()});

  auto scores = log_assignment_[last_layer]->forward(desc0, desc1);
  auto [m0, m1, mscores0, mscores1] =
      matcher::utils::filter_matches(scores, config_.filter_threshold);
  torch::Tensor m_indices_0, m_indices_1;

  if (do_point_pruning) {
    // Get the actual number of matches from m0
    int64_t num_matches = m0.size(1);  // Should be 1708

    // Create batch indices tensor and repeat for each match
    auto batch_indices =
        torch::arange(b, torch::TensorOptions().device(device_));
    m_indices_0 =
        batch_indices.unsqueeze(1).expand({b, num_matches}).reshape(-1);

    // Flatten match indices and create mask for valid matches
    m_indices_1 = m0.reshape(-1);
    auto valid_mask = m_indices_1 >= 0;

    // Apply mask to both tensors using masked_select
    m_indices_0 = m_indices_0.masked_select(valid_mask);
    m_indices_1 = m_indices_1.masked_select(valid_mask);

    // Use advanced indexing to select final indices
    if (m_indices_0.numel() > 0 && m_indices_1.numel() > 0) {
      m_indices_0 = ind0.index({torch::indexing::Slice(), m_indices_0});
      m_indices_1 = ind1.index({torch::indexing::Slice(), m_indices_1});
    }
  } else {
    torch::Tensor m_indices_0 =
        torch::empty({0}, torch::TensorOptions().device(device_));
    torch::Tensor m_indices_1 =
        torch::empty({0}, torch::TensorOptions().device(device_));
  }

  auto matches = torch::stack({m_indices_0, m_indices_1}, 0);

  // Update m0, m1, mscores tensors
  if (do_point_pruning) {
    auto m0_ = torch::full(
        {b, m}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto m1_ = torch::full(
        {b, n}, -1, torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto mscores0_ = torch::zeros({b, m}, device_);
    auto mscores1_ = torch::zeros({b, n}, device_);

    m0_.index_put_({torch::indexing::Slice(), ind0},
                   torch::where(m0 == -1, -1, ind1.gather(1, m0.clamp(0))));
    m1_.index_put_({torch::indexing::Slice(), ind1},
                   torch::where(m1 == -1, -1, ind0.gather(1, m1.clamp(0))));

    mscores0_.index_put_({torch::indexing::Slice(), ind0}, mscores0);
    mscores1_.index_put_({torch::indexing::Slice(), ind1}, mscores1);

    m0 = m0_;
    m1 = m1_;
    mscores0 = mscores0_;
    mscores1 = mscores1_;
  } else {
    prune0 = torch::ones_like(mscores0) * config_.n_layers;
    prune1 = torch::ones_like(mscores1) * config_.n_layers;
  }

  // Prepare output
  torch::Dict<std::string, torch::Tensor> output;
  output.insert("matches0", m0);
  output.insert("matches1", m1);
  output.insert("matching_scores0", mscores0);
  output.insert("matching_scores1", mscores1);
  output.insert("matches", matches);
  output.insert("stop", torch::tensor(last_layer + 1));
  output.insert("prune0", prune0);
  output.insert("prune1", prune1);

  return output;
}

void LightGlue::load_parameters(const std::string& model_path) {
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
  for (const auto& p : model_params) {
    param_map.emplace(p.key(), p.value());
  }

  // Collect buffer names
  for (const auto& b : model_buffers) {
    buffer_map.emplace(b.key(), b.value());
  }

  // Update parameters and buffers
  torch::NoGradGuard no_grad;

  for (const auto& w : weights) {
    const auto name = map_python_to_cpp(w.key().toStringRef());

    const auto& param = w.value().toTensor();

    // Try parameters first
    if (auto it = param_map.find(name); it != param_map.end()) {
      if (it->second.sizes() == param.sizes()) {
        it->second.copy_(param);
      } else {
        throw std::runtime_error(
            "Shape mismatch for parameter: " + name +
            " Expected: " + std::to_string(it->second.numel()) +
            " Got: " + std::to_string(param.numel()));
      }
      continue;
    }

    // Then try buffers
    if (auto it = buffer_map.find(name); it != buffer_map.end()) {
      if (it->second.sizes() == param.sizes()) {
        it->second.copy_(param);
      } else {
        std::cout << "buffer name: " << name
                  << "Expected: " << it->second.sizes()
                  << ", Got: " << param.sizes() << std::endl;
        throw std::runtime_error(
            "Shape mismatch for buffer: " + name +
            " Expected: " + std::to_string(it->second.numel()) +
            " Got: " + std::to_string(param.numel()));
      }
      continue;
    }

    if (name == "confidence_thresholds") {
      // Confidence thresholds are not in the parameters but hard-coded above.
      continue;
    }

    // Parameter not found in model
    std::cerr << "Warning: " << name
              << " not found in model parameters or buffers\n";
  }
}

std::vector<char> LightGlue::get_the_bytes(const std::string& filename) {
  // Use RAII file handling
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
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

  while (file.read(chunk, CHUNK_SIZE)) {
    buffer.insert(buffer.end(), chunk, chunk + file.gcount());
  }
  if (file.gcount() > 0) {
    buffer.insert(buffer.end(), chunk, chunk + file.gcount());
  }

  if (buffer.size() != size) {
    std::ostringstream error;
    error << "Failed to read expected number of bytes: " << buffer.size() << " vs " << size;
    throw std::runtime_error(error.str());
  }

  return buffer;
}

}  // namespace matcher
