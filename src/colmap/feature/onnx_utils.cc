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

#include "colmap/feature/onnx_utils.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <mutex>
#include <sstream>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace colmap {

#ifdef COLMAP_ONNX_ENABLED

std::string FormatONNXTensorShape(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

void ThrowCheckONNXNode(const std::string_view name,
                        const std::string_view expected_name,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& expected_shape) {
  THROW_CHECK_EQ(name, expected_name);
  THROW_CHECK_EQ(shape.size(), expected_shape.size())
      << "Invalid shape for " << name << ": " << FormatONNXTensorShape(shape)
      << " != " << FormatONNXTensorShape(expected_shape);
  for (size_t i = 0; i < shape.size(); ++i) {
    // -1 is treated as a wildcard for dynamic dimensions.
    if (expected_shape[i] != -1) {
      THROW_CHECK_EQ(shape[i], expected_shape[i])
          << "Invalid shape for " << name << ": "
          << FormatONNXTensorShape(shape)
          << " != " << FormatONNXTensorShape(expected_shape);
    }
  }
}

ONNXModel::ONNXModel(std::string model_path,
                     int num_threads,
                     bool use_gpu,
                     const std::string& gpu_index) {
  {
    static std::mutex download_mutex;
    const std::lock_guard<std::mutex> lock(download_mutex);
    model_path = MaybeDownloadAndCacheFile(model_path).string();
  }

  const int num_eff_threads = GetEffectiveNumThreads(num_threads);
  session_options_.SetInterOpNumThreads(num_eff_threads);
  session_options_.SetIntraOpNumThreads(num_eff_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

#ifdef COLMAP_CUDA_ENABLED
  if (use_gpu) {
    const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "ONNX model can only run on one GPU";
    OrtCUDAProviderOptions cuda_options{};
    if (gpu_indices[0] >= 0) {
      cuda_options.device_id = gpu_indices[0];
    }
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }
#endif

  VLOG(2) << "Loading ONNX model from " << model_path;
#ifdef _WIN32
  constexpr int kCodePage = CP_UTF8;
  const int wide_len =
      MultiByteToWideChar(kCodePage, 0, model_path.c_str(), -1, nullptr, 0);
  std::wstring model_path_wide(wide_len, L'\0');
  MultiByteToWideChar(
      kCodePage, 0, model_path.c_str(), -1, &model_path_wide[0], wide_len);
  const wchar_t* model_path_cstr = model_path_wide.c_str();
#else
  const char* model_path_cstr = model_path.c_str();
#endif
  session_ =
      std::make_unique<Ort::Session>(env_, model_path_cstr, session_options_);

  VLOG(2) << "Parsing the inputs";
  const int num_inputs = session_->GetInputCount();
  input_name_strs_.reserve(num_inputs);
  input_names_.reserve(num_inputs);
  input_shapes_.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_name_strs_.emplace_back(
        session_->GetInputNameAllocated(i, allocator_));
    input_names_.emplace_back(input_name_strs_[i].get());
    input_shapes_.emplace_back(
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  VLOG(2) << "Parsing the outputs";
  const int num_outputs = session_->GetOutputCount();
  output_name_strs_.reserve(num_outputs);
  output_names_.reserve(num_outputs);
  output_shapes_.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_name_strs_.emplace_back(
        session_->GetOutputNameAllocated(i, allocator_));
    output_names_.emplace_back(output_name_strs_[i].get());
    output_shapes_.emplace_back(
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

std::vector<Ort::Value> ONNXModel::Run(
    const std::vector<Ort::Value>& input_tensors) const {
  return session_->Run(Ort::RunOptions(),
                       input_names_.data(),
                       input_tensors.data(),
                       input_tensors.size(),
                       output_names_.data(),
                       output_names_.size());
}

#endif  // COLMAP_ONNX_ENABLED

}  // namespace colmap
