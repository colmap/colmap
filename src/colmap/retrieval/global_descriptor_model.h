// SPDX-License-Identifier: BSD-3-Clause

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
