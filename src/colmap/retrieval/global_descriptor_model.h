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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace colmap {
namespace retrieval {

// Configuration for a global descriptor image retrieval model.
// Each model defines its preprocessing, ONNX I/O shapes, and default
// download URI.  Adding a new model = adding one entry to the registry.
struct GlobalDescriptorModel {
  // Display name (e.g. "MixVPR", "MegaLoc").
  std::string name;

  // Input image size for preprocessing (width, height).
  // Both 0 means "keep original size" (model handles dynamic input).
  int input_width = 320;
  int input_height = 320;

  // Per-channel normalization mean (applied after scaling to [0,1]).
  std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};

  // Per-channel normalization std (applied after scaling to [0,1]).
  std::array<float, 3> std = {0.229f, 0.224f, 0.225f};

  // ONNX I/O names.
  std::string input_name = "images";
  std::string output_name = "descriptor";

  // Expected ONNX input shape (for validation).  -1 = dynamic dimension.
  std::vector<int64_t> expected_input_shape = {-1, 3, 320, 320};

  // Expected ONNX output shape (for validation).  -1 = dynamic dimension.
  std::vector<int64_t> expected_output_shape = {-1, 4096};

  // Dimensionality of the output descriptor.
  int descriptor_dim = 4096;

  // Whether the model architecture supports batched inference.
  bool supports_batching = true;

  // A precision variant of the model with its download URI.
  struct Variant {
    // Lowercase variant key (e.g. "fp16", "fp32").
    std::string name;
    // Download URI in COLMAP format: "url;filename;sha256". Must point to
    // a single-file ONNX export (the downloader does not support ONNX
    // external data). Files follow the "<model>_<variant>.onnx" naming.
    std::string uri;
    // Whether this exported file supports batched inference.
    bool supports_batching = true;
  };

  // Precision variants, default variant first.
  std::vector<Variant> variants;

  // Returns the model config for a given model name.
  // Returns nullptr if the name is not recognized.
  static const GlobalDescriptorModel* GetModel(std::string_view name);

  // Returns all registered model names.
  static std::vector<std::string_view> ModelNames();

  // Returns the model URI for the given model name and precision variant.
  // An empty precision selects the model's default variant. Returns an
  // empty string if the model or variant is not registered.
  static std::string DefaultModelUri(std::string_view name,
                                     std::string_view precision = "");

  // Returns the precision variants of the given model, default first.
  static std::vector<std::string_view> VariantNames(std::string_view name);

  // Returns whether the given model and precision variant support batched
  // inference. Returns false if the model or variant is not registered.
  static bool SupportsBatching(std::string_view name,
                               std::string_view precision = "");
};

}  // namespace retrieval
}  // namespace colmap
