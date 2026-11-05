// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
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
inline Ort::Value CreateONNXTensor(const T* data,
                                   size_t num_elements,
                                   const std::vector<int64_t>& shape) {
  // Ort::Value::CreateTensor takes a non-const pointer even for read-only
  // input tensors; ONNXModel::Run never mutates its input tensors.
  return Ort::Value::CreateTensor<T>(
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                 OrtMemType::OrtMemTypeCPU),
      const_cast<T*>(data),
      num_elements,
      shape.data(),
      shape.size());
}

template <typename T>
inline Ort::Value CreateONNXTensor(const std::vector<T>& data,
                                   const std::vector<int64_t>& shape) {
  return CreateONNXTensor(data.data(), data.size(), shape);
}

template <typename T>
void CreateONNXTensor(const std::vector<T>&&,
                      const std::vector<int64_t>&) = delete;

// Create a scalar (rank-0) CPU ONNX tensor.
template <typename T>
inline Ort::Value CreateONNXScalarTensor(const T& value) {
  static const std::vector<int64_t> kEmptyShape;
  return CreateONNXTensor(&value, 1, kEmptyShape);
}

template <typename T>
void CreateONNXScalarTensor(const T&&) = delete;

// Check that a model node has the expected element type.
void ThrowCheckONNXElementType(std::string_view name,
                               ONNXTensorElementDataType type,
                               ONNXTensorElementDataType expected_type);

// Return whether an ONNX Runtime error message indicates a device / BFC arena
// out-of-memory allocation failure.
bool IsONNXOutOfMemoryError(std::string_view message);

// Adjustable counting semaphore bounding concurrent ONNX Runtime session
// executions across worker threads (and across multiple ONNXModel instances
// belonging to the same matcher, e.g., FlexGlue's descriptor and matcher).
// Also tracks per-thread wait and execution durations and permanently lowers
// its concurrency ceiling if a CUDA out-of-memory error occurs.
class ONNXInferenceLimiter {
 public:
  struct ThreadStats {
    int64_t wait_ns = 0;
    int64_t run_ns = 0;
    int num_runs = 0;
  };

  class Permit {
   public:
    Permit() = default;
    Permit(ONNXInferenceLimiter* limiter,
           int slot,
           int limit_at_acquire,
           int64_t wait_ns,
           std::chrono::steady_clock::time_point acquired_at);
    Permit(Permit&& other) noexcept;
    Permit& operator=(Permit&& other) noexcept;
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    ~Permit();

    void Release();
    int slot() const { return slot_; }
    int limit_at_acquire() const { return limit_at_acquire_; }
    int64_t wait_ns() const { return wait_ns_; }

   private:
    ONNXInferenceLimiter* limiter_ = nullptr;
    int slot_ = 0;
    int limit_at_acquire_ = 1;
    int64_t wait_ns_ = 0;
    std::chrono::steady_clock::time_point acquired_at_{};
  };

  explicit ONNXInferenceLimiter(int max_concurrent = 1);

  Permit Acquire();
  void SetMaxConcurrent(int max_concurrent);
  int MaxConcurrent() const;
  int ActiveCount() const;
  int PeakActiveCount() const;
  int OomCeiling() const;

  // Record an out-of-memory failure for an inference acquired when the limit
  // was `limit_at_acquire`, permanently capping MaxConcurrent() at
  // max(1, limit_at_acquire - 1).
  void OnOutOfMemory(int limit_at_acquire);

  // Atomically read and reset the calling thread's accumulated wait/run stats.
  static ThreadStats ConsumeThreadStats();

 private:
  void ReleaseSlot(int slot, int64_t wait_ns, int64_t run_ns);

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int max_concurrent_ = 1;
  int oom_ceiling_ = std::numeric_limits<int>::max();
  int active_ = 0;
  int peak_active_ = 0;
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

  std::unique_ptr<Ort::Session> CreateCudaReplicaSession() const;
  Ort::Session& GetOrCreateSession(int slot) const;
  void DropReplicaSession(int slot) const;

  // Apply the common, provider-independent session options (threading,
  // execution mode, graph optimization). Resets any previously appended
  // execution providers, so it can be reused to rebuild session_options_ for a
  // CPU-only fallback.
  void ConfigureSessionOptions(int num_threads);

  template <typename RunFn>
  std::vector<Ort::Value> RunWithLimiter(RunFn&& run_fn) const;

  mutable Ort::Env env_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::SessionOptions session_options_;
  std::string model_path_;
  int num_eff_threads_ = 1;
  bool allow_session_replicas_ = true;
  std::unique_ptr<Ort::Session> primary_session_;
  mutable std::mutex replica_sessions_mutex_;
  mutable std::vector<std::unique_ptr<Ort::Session>> replica_sessions_;
  mutable std::mutex single_session_mutex_;
  ONNXExecutionProvider execution_provider_ = ONNXExecutionProvider::CPU;
  int cuda_device_id_ = 0;
  std::shared_ptr<ONNXInferenceLimiter> inference_limiter_;
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
