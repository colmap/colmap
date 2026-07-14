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

#include <unordered_map>

namespace colmap {
namespace retrieval {

namespace {

// ---------------------------------------------------------------------------
// Model definitions — add new models here.
// ---------------------------------------------------------------------------

#ifdef COLMAP_DOWNLOAD_ENABLED
const GlobalDescriptorModel kModels[] = {
    {
        .name = "MixVPR",
        .input_width = 320,
        .input_height = 320,
        .mean = {0.485f, 0.456f, 0.406f},
        .std = {0.229f, 0.224f, 0.225f},
        .input_name = "images",
        .output_name = "descriptor",
        .expected_input_shape = {-1, 3, 320, 320},
        .expected_output_shape = {-1, 4096},
        .descriptor_dim = 4096,
        .supports_batching = true,
        .default_model_uri =
            "https://huggingface.co/Realcat/image_retrieval_checkpoints/"
            "resolve/main/mixvpr/onnx/mixvpr_fp16.onnx;"
            "mixvpr_fp16.onnx;"
            "2afcfb51cd13ed96b242c80f809dae7a47ccbbaf0a6cd3a31f93e39b45dee311",
    },
    {
        .name = "MegaLoc",
        .input_width = 518,   // DINOv2 ViT-B/14 training resolution
        .input_height = 518,  // square — ONNX reshape assumes H == W
        // DINOv2 uses [0,1] range without ImageNet normalization.
        .mean = {0.0f, 0.0f, 0.0f},
        .std = {1.0f, 1.0f, 1.0f},
        .input_name = "images",
        .output_name = "descriptor",
        .expected_input_shape = {-1, 3, -1, -1},  // dynamic H,W; our code resizes to 518×518
        .expected_output_shape = {1, 8448},
        .descriptor_dim = 8448,
        .supports_batching = false,  // gemm_input_reshape hardcodes batch=1
        .default_model_uri =
            "https://huggingface.co/Realcat/image_retrieval_checkpoints/"
            "resolve/main/megaloc/megaloc_fp16.onnx;"
            "megaloc_fp16.onnx;"
            "f7dc23122d301a6173576fcf6952b9e962f967671b115ec418cf5ebd23de87ac",
    },
};
#else
const GlobalDescriptorModel kModels[] = {};
#endif

// Build a name→model lookup map.
auto& ModelMap() {
  static std::unordered_map<std::string_view, const GlobalDescriptorModel*> map;
  if (map.empty()) {
    for (const auto& m : kModels) {
      map[m.name] = &m;
    }
  }
  return map;
}

}  // namespace

const GlobalDescriptorModel* GlobalDescriptorModel::GetModel(
    std::string_view name) {
  auto& map = ModelMap();
  auto it = map.find(name);
  return it != map.end() ? it->second : nullptr;
}

std::vector<std::string_view> GlobalDescriptorModel::ModelNames() {
  std::vector<std::string_view> names;
  names.reserve(ModelMap().size());
  for (const auto& [name, _] : ModelMap()) {
    names.push_back(name);
  }
  return names;
}

std::string GlobalDescriptorModel::Label() const {
  return name + " (" + std::to_string(descriptor_dim) + "d" +
         (supports_batching ? ", batch" : ", single") + ")";
}

}  // namespace retrieval
}  // namespace colmap
