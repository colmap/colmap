// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/matcher.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace colmap {

// Compute the worker thread-pool size per GPU for an ONNX matcher: 1 on CPU or
// CoreML, `options.num_workers_per_gpu` if explicitly set (> 0), or an
// automatic cap bounded by logical CPU cores and available host memory.
int ComputeMaxONNXWorkersPerGpu(const FeatureMatchingOptions& options,
                                int num_gpus = 1);

#ifdef COLMAP_ONNX_ENABLED

class ONNXInferenceLimiter;

// Online controller adapting the number of active CPU preprocessing workers
// N(t) and concurrent GPU model inferences M(t) for a shared ONNX matcher.
class ONNXMatchScheduler {
 public:
  struct CacheVramUsage {
    size_t allocated_bytes = 0;
    size_t unallocated_bytes = 0;
  };

  struct Config {
    bool adaptive_gpu = false;
    int cuda_device_id = 0;
    int fixed_workers = -1;
    int fixed_inferences = -1;
    int max_workers_cap = 8;
    int max_num_keypoints = 32768;
    size_t ram_per_worker_bytes = 32ULL << 20;
    size_t min_slot_vram_bytes = 384ULL << 20;
    std::function<std::optional<size_t>()> avail_ram_fn;
    std::function<bool(size_t* free_bytes, size_t* total_bytes)> avail_vram_fn;
    std::function<CacheVramUsage()> cache_vram_fn;
  };

  class WorkerSlot {
   public:
    WorkerSlot() = default;
    explicit WorkerSlot(ONNXMatchScheduler* scheduler)
        : scheduler_(scheduler) {}
    WorkerSlot(WorkerSlot&& other) noexcept : scheduler_(other.scheduler_) {
      other.scheduler_ = nullptr;
    }
    WorkerSlot& operator=(WorkerSlot&& other) noexcept;
    WorkerSlot(const WorkerSlot&) = delete;
    WorkerSlot& operator=(const WorkerSlot&) = delete;
    ~WorkerSlot() { Release(); }
    void Release();

   private:
    ONNXMatchScheduler* scheduler_ = nullptr;
  };

  static Config DefaultConfig(const FeatureMatchingOptions& options);

  explicit ONNXMatchScheduler(Config config);

  WorkerSlot AcquireWorkerSlot();

  void RecordPair(int64_t prep_ns,
                  int64_t inf_ns,
                  int num_runs,
                  int pair_max_keypoints);

  int ActiveWorkersLimit() const;
  int MaxConcurrentInferences() const;
  const std::shared_ptr<ONNXInferenceLimiter>& inference_limiter() const {
    return inference_limiter_;
  }

 private:
  void ReleaseWorkerSlot();

  Config config_;
  std::shared_ptr<ONNXInferenceLimiter> inference_limiter_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int active_workers_ = 0;
  int active_workers_limit_ = 1;
  int cached_ram_workers_limit_ = 8;

  bool has_vram_baseline_ = false;
  size_t vram_baseline_free_ = 0;
  size_t vram_min_free_ = 0;

  int64_t pairs_observed_ = 0;
  int64_t pairs_at_current_m_ = 0;
  int max_seen_keypoints_ = 0;
  double ema_prep_ns_ = 0.0;
  double ema_inf_ns_ = 0.0;
};

// Base class shared by all ONNX feature matchers, gating active worker count
// N(t), measuring per-pair preprocessing vs. inference timings, and adapting
// (N(t), M(t)) online via ONNXMatchScheduler.
class ONNXFeatureMatcher : public FeatureMatcher {
 public:
  explicit ONNXFeatureMatcher(const FeatureMatchingOptions& options);

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) final;

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override;

 protected:
  virtual void MatchImpl(const Image& image1,
                         const Image& image2,
                         FeatureMatches* matches) = 0;

  virtual ONNXMatchScheduler::CacheVramUsage GetCacheVramUsage() const {
    return {};
  }

  const std::shared_ptr<ONNXInferenceLimiter>& inference_limiter() const {
    return scheduler_.inference_limiter();
  }

 private:
  FeatureMatcherType type_;
  ONNXMatchScheduler scheduler_;
};

#endif  // COLMAP_ONNX_ENABLED

struct BruteForceONNXMatchingOptions {
  // The minimum cosine similarity for a match to be considered valid
  // in brute-force matching.
  double min_cossim = 0.85;

  // Maximum ratio for Lowe's ratio test (second-best / best distance).
  double max_ratio = 1.0;

  // Enable cross-checking (mutual nearest neighbor).
  bool cross_check = true;

  // The path to the ONNX model file for the brute-force matcher.
  std::string model_path;

  bool Check() const;
};

// The matcher rejects descriptors whose type is not listed in
// `supported_feature_types`. Set `normalize_descriptors` when the extractor
// does not produce unit-length descriptors.
std::unique_ptr<FeatureMatcher> CreateBruteForceONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const BruteForceONNXMatchingOptions& brute_force_options,
    std::vector<FeatureExtractorType> supported_feature_types,
    bool normalize_descriptors);

}  // namespace colmap
