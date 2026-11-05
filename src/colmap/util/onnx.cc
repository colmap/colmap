// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/onnx.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef COLMAP_COREML_ENABLED
#include <coreml_provider_factory.h>
#endif

namespace colmap {

#ifdef COLMAP_ONNX_ENABLED

namespace {
[[noreturn]] void RethrowONNXException() {
  try {
    std::rethrow_exception(std::current_exception());
  } catch (const Ort::Exception& e) {
    // ONNX Runtime may write to stderr without a trailing newline.
    // Insert a newline here to avoid mixing with COLMAP log output.
    std::cerr << '\n';
    LOG(ERROR) << "ONNX Runtime error: " << e.what();
    throw;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Unexpected exception during ONNX execution: " << e.what();
    throw;
  } catch (...) {
    LOG(ERROR) << "Unknown exception during ONNX execution";
    throw;
  }
}

const char* FormatONNXElementType(ONNXTensorElementDataType type) {
  // Names in `TensorProto::DataType` order. The enum is auto-numbered from
  // zero and only ever extended by appending, so indexing is stable across
  // runtime versions; entries past an older runtime's maximum are simply
  // unreachable there.
  static constexpr const char* kNames[] = {
      "undefined",      "float",        "uint8",          "int8",
      "uint16",         "int16",        "int32",          "int64",
      "string",         "bool",         "float16",        "double",
      "uint32",         "uint64",       "complex64",      "complex128",
      "bfloat16",       "float8e4m3fn", "float8e4m3fnuz", "float8e5m2",
      "float8e5m2fnuz", "uint4",        "int4",           "float4e2m1",
      "uint2",          "int2",         "float8e8m0"};
  if (type < 0 || static_cast<size_t>(type) >= std::size(kNames)) {
    return "unknown";
  }
  return kNames[type];
}

thread_local ONNXInferenceLimiter::ThreadStats g_thread_stats;

}  // namespace

ONNXExecutionProvider SelectONNXExecutionProvider(bool use_gpu) {
  if (!use_gpu) {
    return ONNXExecutionProvider::CPU;
  }
#ifdef COLMAP_CUDA_ENABLED
  return ONNXExecutionProvider::CUDA;
#elif defined(COLMAP_COREML_ENABLED)
  return ONNXExecutionProvider::COREML;
#else
  return ONNXExecutionProvider::CPU;
#endif
}

int SelectONNXCudaDevice(const std::string& gpu_index) {
  const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1) << "ONNX model can only run on one GPU";
  return std::max(0, gpu_indices[0]);
}

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

void ThrowCheckONNXElementType(const std::string_view name,
                               const ONNXTensorElementDataType type,
                               const ONNXTensorElementDataType expected_type) {
  THROW_CHECK_EQ(type, expected_type)
      << "Invalid element type for " << name << ": "
      << FormatONNXElementType(type)
      << " != " << FormatONNXElementType(expected_type);
}

bool IsONNXOutOfMemoryError(const std::string_view message) {
  return message.find("Failed to allocate memory") != std::string_view::npos ||
         message.find("out of memory") != std::string_view::npos ||
         message.find("Out of memory") != std::string_view::npos ||
         message.find("cudaErrorMemoryAllocation") != std::string_view::npos;
}

ONNXInferenceLimiter::Permit::Permit(
    ONNXInferenceLimiter* limiter,
    const int slot,
    const int limit_at_acquire,
    const int64_t wait_ns,
    const std::chrono::steady_clock::time_point acquired_at)
    : limiter_(limiter),
      slot_(slot),
      limit_at_acquire_(limit_at_acquire),
      wait_ns_(wait_ns),
      acquired_at_(acquired_at) {}

ONNXInferenceLimiter::Permit::Permit(Permit&& other) noexcept
    : limiter_(other.limiter_),
      slot_(other.slot_),
      limit_at_acquire_(other.limit_at_acquire_),
      wait_ns_(other.wait_ns_),
      acquired_at_(other.acquired_at_) {
  other.limiter_ = nullptr;
}

ONNXInferenceLimiter::Permit& ONNXInferenceLimiter::Permit::operator=(
    Permit&& other) noexcept {
  if (this != &other) {
    Release();
    limiter_ = other.limiter_;
    slot_ = other.slot_;
    limit_at_acquire_ = other.limit_at_acquire_;
    wait_ns_ = other.wait_ns_;
    acquired_at_ = other.acquired_at_;
    other.limiter_ = nullptr;
  }
  return *this;
}

ONNXInferenceLimiter::Permit::~Permit() { Release(); }

void ONNXInferenceLimiter::Permit::Release() {
  if (limiter_ == nullptr) {
    return;
  }
  const int64_t run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - acquired_at_)
                             .count();
  ONNXInferenceLimiter* limiter = limiter_;
  const int slot = slot_;
  limiter_ = nullptr;
  limiter->ReleaseSlot(slot, wait_ns_, run_ns);
}

ONNXInferenceLimiter::ONNXInferenceLimiter(const int max_concurrent)
    : max_concurrent_(max_concurrent), slot_in_use_(max_concurrent, 0) {
  THROW_CHECK_GE(max_concurrent_, 1);
}

ONNXInferenceLimiter::Permit ONNXInferenceLimiter::Acquire() {
  const auto wait_start = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() { return active_ < max_concurrent_; });
  ++active_;
  peak_active_ = std::max(peak_active_, active_);
  const int limit_at_acquire = max_concurrent_;
  if (static_cast<int>(slot_in_use_.size()) < limit_at_acquire) {
    slot_in_use_.resize(limit_at_acquire, 0);
  }
  int slot = 0;
  while (slot < static_cast<int>(slot_in_use_.size()) &&
         slot_in_use_[slot] != 0) {
    ++slot;
  }
  if (slot >= static_cast<int>(slot_in_use_.size())) {
    slot_in_use_.resize(slot + 1, 0);
  }
  slot_in_use_[slot] = 1;
  lock.unlock();
  const auto acquired_at = std::chrono::steady_clock::now();
  const int64_t wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              acquired_at - wait_start)
                              .count();
  return Permit(this, slot, limit_at_acquire, wait_ns, acquired_at);
}

void ONNXInferenceLimiter::SetMaxConcurrent(const int max_concurrent) {
  THROW_CHECK_GE(max_concurrent, 1);
  bool increased = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const int clamped = std::min(max_concurrent, oom_ceiling_);
    increased = clamped > max_concurrent_;
    max_concurrent_ = clamped;
  }
  if (increased) {
    cv_.notify_all();
  }
}

int ONNXInferenceLimiter::MaxConcurrent() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return max_concurrent_;
}

int ONNXInferenceLimiter::ActiveCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

int ONNXInferenceLimiter::PeakActiveCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return peak_active_;
}

int ONNXInferenceLimiter::OomCeiling() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return oom_ceiling_;
}

void ONNXInferenceLimiter::OnOutOfMemory(const int limit_at_acquire) {
  std::lock_guard<std::mutex> lock(mutex_);
  oom_ceiling_ = std::min(oom_ceiling_, std::max(1, limit_at_acquire - 1));
  max_concurrent_ = std::min(max_concurrent_, oom_ceiling_);
}

ONNXInferenceLimiter::ThreadStats ONNXInferenceLimiter::ConsumeThreadStats() {
  const ThreadStats stats = g_thread_stats;
  g_thread_stats = ThreadStats{};
  return stats;
}

void ONNXInferenceLimiter::ReleaseSlot(const int slot,
                                       const int64_t wait_ns,
                                       const int64_t run_ns) {
  g_thread_stats.wait_ns += wait_ns;
  g_thread_stats.run_ns += run_ns;
  g_thread_stats.num_runs += 1;

  bool should_notify = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= 0 && slot < static_cast<int>(slot_in_use_.size())) {
      slot_in_use_[slot] = 0;
    }
    --active_;
    should_notify = active_ < max_concurrent_;
  }
  if (should_notify) {
    cv_.notify_one();
  }
}

namespace {

#ifdef COLMAP_CUDA_ENABLED
void AppendCudaProviderOptions(Ort::SessionOptions* session_options,
                               const int cuda_device_id) {
  const OrtApi& api = Ort::GetApi();
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  struct CudaOptionsGuard {
    const OrtApi& api;
    OrtCUDAProviderOptionsV2* opts;
    ~CudaOptionsGuard() { api.ReleaseCUDAProviderOptions(opts); }
  } guard{api, cuda_options};

  const std::string device_id_str = std::to_string(cuda_device_id);
  // kSameAsRequested grows the CUDA BFC arena by the requested buffer size
  // instead of doubling to powers of two; use_ep_level_unified_stream binds one
  // CUDA stream + cuBLAS/cuDNN context per Ort::Session rather than allocating
  // duplicate contexts per calling worker thread.
  const char* keys[] = {
      "device_id", "arena_extend_strategy", "use_ep_level_unified_stream"};
  const char* values[] = {device_id_str.c_str(), "kSameAsRequested", "1"};
  Ort::ThrowOnError(
      api.UpdateCUDAProviderOptions(cuda_options, keys, values, 3));
  session_options->AppendExecutionProvider_CUDA_V2(*cuda_options);
}
#endif

std::unique_ptr<Ort::Session> CreateOrtSession(
    Ort::Env& env,
    const std::string& model_path,
    const Ort::SessionOptions& session_options) {
#ifdef _WIN32
  const unsigned int code_page = GetACP();
  const int wide_len =
      MultiByteToWideChar(code_page, 0, model_path.c_str(), -1, nullptr, 0);
  std::wstring model_path_wide(wide_len, L'\0');
  MultiByteToWideChar(
      code_page, 0, model_path.c_str(), -1, &model_path_wide[0], wide_len);
  return std::make_unique<Ort::Session>(
      env, model_path_wide.c_str(), session_options);
#else
  return std::make_unique<Ort::Session>(
      env, model_path.c_str(), session_options);
#endif
}

}  // namespace

ONNXModel::ONNXModel(const std::string& model_path,
                     int num_threads,
                     bool use_gpu,
                     const std::string& gpu_index,
                     std::shared_ptr<ONNXInferenceLimiter> inference_limiter,
                     const bool allow_session_replicas)
    : allow_session_replicas_(allow_session_replicas),
      inference_limiter_(std::move(inference_limiter)) {
  {
    static std::mutex download_mutex;
    const std::lock_guard<std::mutex> lock(download_mutex);
    model_path_ = MaybeDownloadAndCacheFile(model_path).string();
  }

  num_eff_threads_ = GetEffectiveNumThreads(num_threads);

  try {
    InitializeSession(model_path_, num_eff_threads_, use_gpu, gpu_index);
  } catch (...) {
    RethrowONNXException();
  }
}

void ONNXModel::ConfigureSessionOptions(int num_threads) {
  // Reset any previously appended execution providers (e.g. when rebuilding for
  // a CPU-only fallback after an accelerator fails to initialize).
  session_options_ = Ort::SessionOptions();

  // Use sequential execution mode with a single inter-op thread, since our
  // models (ALIKED, LightGlue) are sequential CNNs/Transformers without
  // independent graph branches. Inter-op parallelism would only cause thread
  // contention. Intra-op threads parallelize within individual operators
  // (convolutions, matrix multiplications) and are managed at the caller level.
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetExecutionMode(ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
}

void ONNXModel::InitializeSession(const std::string& model_path,
                                  int num_threads,
                                  bool use_gpu,
                                  const std::string& gpu_index) {
  ConfigureSessionOptions(num_threads);
  execution_provider_ = SelectONNXExecutionProvider(use_gpu);

#ifdef COLMAP_CUDA_ENABLED
  if (execution_provider_ == ONNXExecutionProvider::CUDA) {
    cuda_device_id_ = SelectONNXCudaDevice(gpu_index);
    AppendCudaProviderOptions(&session_options_, cuda_device_id_);
  }
#endif

  // On Apple platforms there is no CUDA, so map use_gpu onto CoreML, which
  // offloads supported subgraphs to the GPU/Apple Neural Engine (unsupported
  // nodes automatically fall back to the CPU). Selected automatically whenever
  // GPU use is requested; set use_gpu=false to force pure CPU execution.
  const bool use_coreml = execution_provider_ == ONNXExecutionProvider::COREML;
#ifdef COLMAP_COREML_ENABLED
  if (use_coreml) {
    VLOG(2) << "Enabling CoreML execution provider";
    // COREML_FLAG_CREATE_MLPROGRAM selects the newer ML Program model format,
    // which supports a wider set of operators and float inputs than the legacy
    // NeuralNetwork format.
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
        static_cast<OrtSessionOptions*>(session_options_),
        COREML_FLAG_CREATE_MLPROGRAM));
  }
#endif

  VLOG(2) << "Loading ONNX model from " << model_path;
  try {
    primary_session_ = CreateOrtSession(env_, model_path, session_options_);
  } catch (const Ort::Exception& e) {
    if (!use_coreml) {
      throw;
    }
    // Some models cannot be compiled by CoreML (e.g. unsupported dynamic
    // shapes). Rather than failing, fall back to pure CPU execution.
    LOG(WARNING) << "Failed to initialize ONNX session with CoreML ("
                 << e.what() << "); falling back to CPU execution provider";
    ConfigureSessionOptions(num_threads);
    execution_provider_ = ONNXExecutionProvider::CPU;
    primary_session_ = CreateOrtSession(env_, model_path, session_options_);
  }

  VLOG(2) << "Parsing the inputs";
  const int num_inputs = primary_session_->GetInputCount();
  input_name_strs_.reserve(num_inputs);
  input_names_.reserve(num_inputs);
  input_shapes_.reserve(num_inputs);
  input_element_types_.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_name_strs_.emplace_back(
        primary_session_->GetInputNameAllocated(i, allocator_));
    input_names_.emplace_back(input_name_strs_[i].get());
    input_shapes_.emplace_back(primary_session_->GetInputTypeInfo(i)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape());
    input_element_types_.emplace_back(primary_session_->GetInputTypeInfo(i)
                                          .GetTensorTypeAndShapeInfo()
                                          .GetElementType());
  }

  VLOG(2) << "Parsing the outputs";
  const int num_outputs = primary_session_->GetOutputCount();
  output_name_strs_.reserve(num_outputs);
  output_names_.reserve(num_outputs);
  output_shapes_.reserve(num_outputs);
  output_element_types_.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_name_strs_.emplace_back(
        primary_session_->GetOutputNameAllocated(i, allocator_));
    output_names_.emplace_back(output_name_strs_[i].get());
    output_shapes_.emplace_back(primary_session_->GetOutputTypeInfo(i)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape());
    output_element_types_.emplace_back(primary_session_->GetOutputTypeInfo(i)
                                           .GetTensorTypeAndShapeInfo()
                                           .GetElementType());
  }
}

std::unique_ptr<Ort::Session> ONNXModel::CreateCudaReplicaSession() const {
  Ort::SessionOptions opts;
  opts.SetInterOpNumThreads(1);
  opts.SetIntraOpNumThreads(num_eff_threads_);
  opts.SetExecutionMode(ORT_SEQUENTIAL);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
#ifdef COLMAP_CUDA_ENABLED
  AppendCudaProviderOptions(&opts, cuda_device_id_);
#endif
  return CreateOrtSession(env_, model_path_, opts);
}

Ort::Session& ONNXModel::GetOrCreateSession(const int slot) const {
  if (slot <= 0 || execution_provider_ != ONNXExecutionProvider::CUDA ||
      !allow_session_replicas_) {
    return *primary_session_;
  }
  const int replica_idx = slot - 1;
  std::lock_guard<std::mutex> lock(replica_sessions_mutex_);
  if (replica_idx < static_cast<int>(replica_sessions_.size()) &&
      replica_sessions_[replica_idx] != nullptr) {
    return *replica_sessions_[replica_idx];
  }
  if (static_cast<int>(replica_sessions_.size()) <= replica_idx) {
    replica_sessions_.resize(replica_idx + 1);
  }
  replica_sessions_[replica_idx] = CreateCudaReplicaSession();
  return *replica_sessions_[replica_idx];
}

void ONNXModel::DropReplicaSession(const int slot) const {
  if (slot <= 0) {
    return;
  }
  const int replica_idx = slot - 1;
  std::lock_guard<std::mutex> lock(replica_sessions_mutex_);
  if (replica_idx < static_cast<int>(replica_sessions_.size())) {
    replica_sessions_[replica_idx].reset();
  }
}

template <typename RunFn>
std::vector<Ort::Value> ONNXModel::RunWithLimiter(RunFn&& run_fn) const {
  try {
    ONNXInferenceLimiter::Permit permit = inference_limiter_ != nullptr
                                              ? inference_limiter_->Acquire()
                                              : ONNXInferenceLimiter::Permit();
    const int limit_at_acquire = permit.limit_at_acquire();
    const int slot = permit.slot();
    try {
      if (!allow_session_replicas_ && inference_limiter_ != nullptr &&
          execution_provider_ == ONNXExecutionProvider::CUDA) {
        std::lock_guard<std::mutex> lock(single_session_mutex_);
        return run_fn(*primary_session_, Ort::RunOptions());
      }
      return run_fn(GetOrCreateSession(slot), Ort::RunOptions());
    } catch (const Ort::Exception& e) {
      if (execution_provider_ != ONNXExecutionProvider::CUDA ||
          !IsONNXOutOfMemoryError(e.what())) {
        throw;
      }
      if (inference_limiter_ != nullptr) {
        inference_limiter_->OnOutOfMemory(limit_at_acquire);
      }
      DropReplicaSession(slot);
      permit.Release();
      LOG(WARNING) << "ONNX CUDA out-of-memory (" << e.what()
                   << "); shrinking GPU memory arena and retrying";
      ONNXInferenceLimiter::Permit retry_permit =
          inference_limiter_ != nullptr ? inference_limiter_->Acquire()
                                        : ONNXInferenceLimiter::Permit();
      Ort::RunOptions shrink_options;
      const std::string gpu_mem = "gpu:" + std::to_string(cuda_device_id_);
      shrink_options.AddConfigEntry("memory.enable_memory_arena_shrinkage",
                                    gpu_mem.c_str());
      if (!allow_session_replicas_ && inference_limiter_ != nullptr) {
        std::lock_guard<std::mutex> lock(single_session_mutex_);
        return run_fn(*primary_session_, shrink_options);
      }
      return run_fn(GetOrCreateSession(retry_permit.slot()), shrink_options);
    }
  } catch (...) {
    RethrowONNXException();
  }
}

std::vector<Ort::Value> ONNXModel::Run(
    const std::vector<Ort::Value>& input_tensors) const {
  return RunWithLimiter(
      [&](Ort::Session& session, const Ort::RunOptions& run_options) {
        return session.Run(run_options,
                           input_names_.data(),
                           input_tensors.data(),
                           input_tensors.size(),
                           output_names_.data(),
                           output_names_.size());
      });
}

std::vector<Ort::Value> ONNXModel::Run(
    const std::vector<Ort::Value>& input_tensors,
    const Ort::MemoryInfo& output_memory_info) const {
  return RunWithLimiter(
      [&](Ort::Session& session, const Ort::RunOptions& run_options) {
        THROW_CHECK_EQ(input_tensors.size(), input_names_.size());
        Ort::IoBinding binding(session);
        for (size_t i = 0; i < input_tensors.size(); ++i) {
          binding.BindInput(input_names_[i], input_tensors[i]);
        }
        for (const char* name : output_names_) {
          binding.BindOutput(name, output_memory_info);
        }
        session.Run(run_options, binding);
        binding.SynchronizeOutputs();
        return binding.GetOutputValues();
      });
}

#endif  // COLMAP_ONNX_ENABLED

}  // namespace colmap
