// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace colmap {
namespace mvs {

// Filled with release URLs and SHA-256 hashes when the converted models are
// published by colmap/mvsformerplusplus-onnx. Empty values intentionally
// require an explicit model_path for development builds.
inline const std::string kDefaultMVSFormerPlusPlus5ViewUri = "";
inline const std::string kDefaultMVSFormerPlusPlus10ViewUri = "";

}  // namespace mvs
}  // namespace colmap
