// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#ifdef COLMAP_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>

namespace colmap {

enum class ONNXExecutionProvider {
  CPU,
  CUDA,
  COREML,
};

// Resolve the execution provider requested by `use_gpu` for this build.
ONNXExecutionProvider SelectONNXExecutionProvider(bool use_gpu);

// Resolve the CUDA device on which an ONNXModel with the given `gpu_index`
// runs. A negative index selects device 0. Throws if more than one GPU is
// selected.
int SelectONNXCudaDevice(const std::string& gpu_index);

// Format tensor shape as a string for logging/error messages.
std::string FormatONNXTensorShape(const std::vector<int64_t>& shape);

// Check that a model node has the expected name and shape.
// Shape values of -1 are treated as wildcards (dynamic dimensions).
void ThrowCheckONNXNode(std::string_view name,
                        std::string_view expected_name,
                        const std::vector<int64_t>& shape,
                        const std::vector<int64_t>& expected_shape);

// Create a CPU ONNX tensor that references the given data. The caller must
// keep the data alive until the tensor is consumed by ONNXModel::Run().
template <typename T>
inline Ort::Value CreateONNXTensor(T* data,
                                   size_t num_elements,
                                   const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<T>(
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                 OrtMemType::OrtMemTypeCPU),
      data,
      num_elements,
      shape.data(),
      shape.size());
}

template <typename T>
inline Ort::Value CreateONNXTensor(std::vector<T>& data,
                                   const std::vector<int64_t>& shape) {
  return CreateONNXTensor(data.data(), data.size(), shape);
}

// Create a scalar (rank-0) CPU ONNX tensor.
template <typename T>
inline Ort::Value CreateONNXScalarTensor(T& value) {
  static const std::vector<int64_t> kEmptyShape;
  return CreateONNXTensor(&value, 1, kEmptyShape);
}

// Check that a model node has the expected element type.
void ThrowCheckONNXElementType(std::string_view name,
                               ONNXTensorElementDataType type,
                               ONNXTensorElementDataType expected_type);

// Counting semaphore that bounds the number of concurrent ONNX model
// inferences (e.g., on a shared GPU) across a larger pool of worker threads,
// assigning each active caller a distinct slot in [0, max_concurrent).
class ONNXInferenceLimiter {
 public:
  class Permit {
   public:
    Permit() = default;
    Permit(Permit&& other) noexcept;
    Permit& operator=(Permit&& other) noexcept;
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    ~Permit();

    void Release();
    int slot() const { return slot_; }

   private:
    friend class ONNXInferenceLimiter;
    Permit(ONNXInferenceLimiter* limiter, int slot);

    ONNXInferenceLimiter* limiter_ = nullptr;
    int slot_ = 0;
  };

  explicit ONNXInferenceLimiter(int max_concurrent);

  Permit Acquire();

  int MaxConcurrent() const { return max_concurrent_; }
  int ActiveCount() const;

 private:
  void ReleaseSlot(int slot);

  const int max_concurrent_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int active_ = 0;
  std::vector<uint8_t> slot_in_use_;
};

// Wrapper for ONNX Runtime session management.
// Handles model loading, input/output shape parsing, and inference.
class ONNXModel {
 public:
  ONNXModel(const std::string& model_path,
            int num_threads,
            bool use_gpu,
            const std::string& gpu_index,
            std::shared_ptr<ONNXInferenceLimiter> inference_limiter = nullptr,
            bool allow_session_replicas = true);

  std::vector<Ort::Value> Run(
      const std::vector<Ort::Value>& input_tensors) const;

  // Same as above, but the outputs are allocated in the given memory, e.g., on
  // the GPU or in page-locked host memory, so that they can be passed to
  // another model without copies.
  std::vector<Ort::Value> Run(const std::vector<Ort::Value>& input_tensors,
                              const Ort::MemoryInfo& output_memory_info) const;

  const std::vector<std::vector<int64_t>>& input_shapes() const {
    return input_shapes_;
  }
  const std::vector<char*>& input_names() const { return input_names_; }
  const std::vector<ONNXTensorElementDataType>& input_element_types() const {
    return input_element_types_;
  }
  const std::vector<std::vector<int64_t>>& output_shapes() const {
    return output_shapes_;
  }
  const std::vector<char*>& output_names() const { return output_names_; }
  const std::vector<ONNXTensorElementDataType>& output_element_types() const {
    return output_element_types_;
  }
  ONNXExecutionProvider execution_provider() const {
    return execution_provider_;
  }

 private:
  void InitializeSession(const std::string& model_path,
                         int num_threads,
                         bool use_gpu,
                         const std::string& gpu_index);

  // Apply the common, provider-independent session options (threading,
  // execution mode, graph optimization). Resets any previously appended
  // execution providers, so it can be reused to rebuild session_options_ for a
  // CPU-only fallback.
  void ConfigureSessionOptions(int num_threads);

  std::unique_ptr<Ort::Session> CreateCudaReplicaSession() const;
  Ort::Session& GetOrCreateSession(int slot) const;

  template <typename RunFn>
  std::vector<Ort::Value> RunWithLimiter(RunFn&& run_fn) const;

  std::string model_path_;
  int num_eff_threads_ = 1;
#ifdef COLMAP_CUDA_ENABLED
  int cuda_device_id_ = 0;
#endif
  bool allow_session_replicas_ = true;
  std::shared_ptr<ONNXInferenceLimiter> inference_limiter_;
  Ort::Env env_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> primary_session_;
  mutable std::mutex replica_sessions_mutex_;
  mutable std::vector<std::unique_ptr<Ort::Session>> replica_sessions_;
  mutable std::mutex single_session_mutex_;
  ONNXExecutionProvider execution_provider_ = ONNXExecutionProvider::CPU;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<Ort::AllocatedStringPtr> input_name_strs_;
  std::vector<char*> input_names_;
  std::vector<ONNXTensorElementDataType> input_element_types_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<Ort::AllocatedStringPtr> output_name_strs_;
  std::vector<char*> output_names_;
  std::vector<ONNXTensorElementDataType> output_element_types_;
};

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED
