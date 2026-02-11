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

#include "colmap/util/logging.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef COLMAP_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>

namespace colmap {

// Format tensor shape as a string for logging/error messages.
std::string FormatONNXTensorShape(const std::vector<int64_t>& shape);

// Check that a model node has the expected name and shape.
// Shape values of -1 are treated as wildcards (dynamic dimensions).
void ThrowCheckONNXNode(std::string_view name,
                        std::string_view expected_name,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& expected_shape);

// Wrapper for ONNX Runtime session management.
// Handles model loading, input/output shape parsing, and inference.
class ONNXModel {
 public:
  ONNXModel(std::string model_path,
            int num_threads,
            bool use_gpu,
            const std::string& gpu_index);

  std::vector<Ort::Value> Run(
      const std::vector<Ort::Value>& input_tensors) const;

  const std::vector<std::vector<int64_t>>& input_shapes() const {
    return input_shapes_;
  }
  const std::vector<char*>& input_names() const { return input_names_; }
  const std::vector<std::vector<int64_t>>& output_shapes() const {
    return output_shapes_;
  }
  const std::vector<char*>& output_names() const { return output_names_; }

 private:
  Ort::Env env_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<Ort::AllocatedStringPtr> input_name_strs_;
  std::vector<char*> input_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<Ort::AllocatedStringPtr> output_name_strs_;
  std::vector<char*> output_names_;
};

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED
