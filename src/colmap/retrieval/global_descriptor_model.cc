// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/retrieval/global_descriptor_model.h"

#include "colmap/util/string.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace colmap {
namespace retrieval {

namespace {

// ---------------------------------------------------------------------------
// Model definitions — add new models here.
// ---------------------------------------------------------------------------

#ifdef COLMAP_ONNX_ENABLED
const std::vector<GlobalDescriptorModel> kModels = [] {
  // The default options correspond to MixVPR.
  GlobalDescriptorModel mixvpr;
  mixvpr.name = "MixVPR";
  mixvpr.variants = {
      {"fp16",
       "https://huggingface.co/Realcat/image_retrieval_checkpoints/resolve/"
       "main/mixvpr/onnx/mixvpr_fp16.onnx;mixvpr_fp16.onnx;"
       "fee89548fdc8066d2464f00d5672363868459c8d6346c33bf2d2aea3b7e13c86"},
      {"fp32",
       "https://huggingface.co/Realcat/image_retrieval_checkpoints/resolve/"
       "main/mixvpr/onnx/mixvpr_fp32.onnx;mixvpr_fp32.onnx;"
       "1ede695b528f99b4d4ad3da940c9fe8d62f07254cbb0871205d669816ee97f47"}};

  GlobalDescriptorModel megaloc;
  megaloc.name = "MegaLoc";
  megaloc.input_width = 322;
  megaloc.input_height = 322;
  megaloc.expected_input_shape = {-1, 3, 322, 322};
  megaloc.expected_output_shape = {-1, 8448};
  megaloc.descriptor_dim = 8448;
  // Weights are stored in fp16, computation is in fp32.
  megaloc.variants = {
      {"fp16",
       "https://huggingface.co/gberton/MegaLoc/resolve/main/megaloc.onnx;"
       "megaloc.onnx;"
       "9304d7d2473b006d2664736b5c56135c8404ff351d4592cfd9dcb4d7e8f12355"}};

  return std::vector<GlobalDescriptorModel>{mixvpr, megaloc};
}();
#else
const std::vector<GlobalDescriptorModel> kModels = {};
#endif

// Build a lowercase-name→model lookup map for case-insensitive matching.
// The canonical spelling remains the registered model name.
auto& ModelMap() {
  static std::unordered_map<std::string, const GlobalDescriptorModel*> map;
  if (map.empty()) {
    for (const auto& m : kModels) {
      std::string key = m.name;
      StringToLower(&key);
      map[std::move(key)] = &m;
    }
  }
  return map;
}

}  // namespace

const GlobalDescriptorModel* GlobalDescriptorModel::GetModel(
    std::string_view name) {
  std::string key(name);
  StringToLower(&key);
  auto& map = ModelMap();
  auto it = map.find(key);
  return it != map.end() ? it->second : nullptr;
}

std::vector<std::string_view> GlobalDescriptorModel::ModelNames() {
  std::vector<std::string_view> names;
  names.reserve(kModels.size());
  for (const auto& m : kModels) {
    names.push_back(m.name);
  }
  return names;
}

namespace {

const GlobalDescriptorModel::Variant* GetVariant(std::string_view name,
                                                 std::string_view precision) {
  const GlobalDescriptorModel* model = GlobalDescriptorModel::GetModel(name);
  if (model == nullptr || model->variants.empty()) {
    return nullptr;
  }
  if (precision.empty()) {
    return &model->variants.front();
  }
  std::string key(precision);
  StringToLower(&key);
  for (const auto& variant : model->variants) {
    if (variant.name == key) {
      return &variant;
    }
  }
  return nullptr;
}

}  // namespace

std::string GlobalDescriptorModel::DefaultModelUri(std::string_view name,
                                                   std::string_view precision) {
  const Variant* variant = GetVariant(name, precision);
  return variant == nullptr ? std::string() : variant->uri;
}

std::vector<std::string_view> GlobalDescriptorModel::VariantNames(
    std::string_view name) {
  const GlobalDescriptorModel* model = GetModel(name);
  std::vector<std::string_view> names;
  if (model != nullptr) {
    names.reserve(model->variants.size());
    for (const auto& variant : model->variants) {
      names.push_back(variant.name);
    }
  }
  return names;
}

bool GlobalDescriptorModel::SupportsBatching(std::string_view name,
                                             std::string_view precision) {
  const GlobalDescriptorModel* model = GetModel(name);
  const Variant* variant = GetVariant(name, precision);
  return model != nullptr && variant != nullptr && model->supports_batching &&
         variant->supports_batching;
}

}  // namespace retrieval
}  // namespace colmap
