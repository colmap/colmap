// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/onnx_matchers.h"

#include "colmap/feature/utils.h"
#include "colmap/util/misc.h"
#include "colmap/util/onnx.h"
#include "colmap/util/threading.h"

#ifdef COLMAP_CUDA_ENABLED
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

namespace colmap {
namespace {

constexpr int kMaxAutoWorkersPerGpu = 8;
constexpr int kMaxAutoInferencesPerGpu = 3;
constexpr size_t kHostRamHeadroomBytes = 512ULL << 20;

size_t RamBytesPerWorker(FeatureMatcherType type) {
  return type == FeatureMatcherType::FLEXGLUE ? (128ULL << 20) : (32ULL << 20);
}

size_t MinSlotVramBytes(FeatureMatcherType type) {
  switch (type) {
    case FeatureMatcherType::FLEXGLUE:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return 1024ULL << 20;
    default:
      return 384ULL << 20;
  }
}

int ComputeRamWorkersCap(const std::optional<size_t>& avail_ram,
                         size_t ram_per_worker_bytes,
                         int num_gpus,
                         int max_cap) {
  if (!avail_ram.has_value()) {
    return max_cap;
  }
  const size_t usable_ram = *avail_ram > kHostRamHeadroomBytes
                                ? (*avail_ram - kHostRamHeadroomBytes)
                                : 0;
  const size_t denom =
      std::max<size_t>(1, ram_per_worker_bytes * std::max(1, num_gpus));
  return std::max<int>(1,
                       static_cast<int>(std::min<size_t>(
                           static_cast<size_t>(max_cap), usable_ram / denom)));
}

}  // namespace

int ComputeMaxONNXWorkersPerGpu(const FeatureMatchingOptions& options,
                                const int num_gpus) {
#ifdef COLMAP_ONNX_ENABLED
  if (!options.SupportsSharedMatcher() ||
      SelectONNXExecutionProvider(options.use_gpu) !=
          ONNXExecutionProvider::CUDA) {
    return 1;
  }
  if (options.num_workers_per_gpu > 0) {
    return options.num_workers_per_gpu;
  }
  const int eff_threads = GetEffectiveNumThreads(options.num_threads);
  const int core_cap =
      std::clamp(eff_threads / std::max(1, num_gpus), 1, kMaxAutoWorkersPerGpu);
  return ComputeRamWorkersCap(GetAvailableMemoryBytes(),
                              RamBytesPerWorker(options.type),
                              num_gpus,
                              core_cap);
#else
  return 1;
#endif
}

#ifdef COLMAP_ONNX_ENABLED

ONNXMatchScheduler::WorkerSlot& ONNXMatchScheduler::WorkerSlot::operator=(
    WorkerSlot&& other) noexcept {
  if (this != &other) {
    Release();
    scheduler_ = other.scheduler_;
    other.scheduler_ = nullptr;
  }
  return *this;
}

void ONNXMatchScheduler::WorkerSlot::Release() {
  if (scheduler_ != nullptr) {
    ONNXMatchScheduler* scheduler = scheduler_;
    scheduler_ = nullptr;
    scheduler->ReleaseWorkerSlot();
  }
}

ONNXMatchScheduler::Config ONNXMatchScheduler::DefaultConfig(
    const FeatureMatchingOptions& options) {
  Config config;
  config.adaptive_gpu = SelectONNXExecutionProvider(options.use_gpu) ==
                        ONNXExecutionProvider::CUDA;
  constexpr int kNominalDetectorKeypoints = 2048;
  config.fixed_workers = options.num_workers_per_gpu;
  config.fixed_inferences = options.num_inferences_per_gpu;
  config.max_num_keypoints =
      std::min(options.max_num_matches, kNominalDetectorKeypoints);
  config.ram_per_worker_bytes = RamBytesPerWorker(options.type);
  config.min_slot_vram_bytes = MinSlotVramBytes(options.type);
  config.avail_ram_fn = []() { return GetAvailableMemoryBytes(); };

  if (config.adaptive_gpu) {
    config.cuda_device_id = SelectONNXCudaDevice(options.gpu_index);
    // FeatureMatcherController sets worker_matching_options.num_threads = 1 to
    // prevent nested ORT intra-op threadpools, while bounding the spawned
    // worker pool per GPU itself; use hardware cores here for standalone use.
    const int hw_threads = GetEffectiveNumThreads(
        options.num_threads > 1 ? options.num_threads : -1);
    config.max_workers_cap =
        options.num_workers_per_gpu > 0
            ? options.num_workers_per_gpu
            : std::clamp(hw_threads, 1, kMaxAutoWorkersPerGpu);
#ifdef COLMAP_CUDA_ENABLED
    const int device_id = config.cuda_device_id;
    config.avail_vram_fn = [device_id](size_t* free_bytes,
                                       size_t* total_bytes) {
      return GetAvailableCudaMemoryBytes(device_id, free_bytes, total_bytes);
    };
#endif
  } else {
    config.max_workers_cap = 1;
  }
  return config;
}

ONNXMatchScheduler::ONNXMatchScheduler(Config config)
    : config_(std::move(config)) {
  config_.max_workers_cap = std::max(1, config_.max_workers_cap);
  cached_ram_workers_limit_ = config_.max_workers_cap;
  if (!config_.adaptive_gpu) {
    active_workers_limit_ = 1;
    inference_limiter_ = std::make_shared<ONNXInferenceLimiter>(1);
    return;
  }

  const int max_workers = config_.fixed_workers > 0 ? config_.fixed_workers
                                                    : config_.max_workers_cap;
  const int initial_m =
      config_.fixed_inferences > 0
          ? std::max(1, std::min(config_.fixed_inferences, max_workers))
          : 1;
  inference_limiter_ = std::make_shared<ONNXInferenceLimiter>(initial_m);

  if (config_.fixed_workers > 0) {
    active_workers_limit_ = config_.fixed_workers;
  } else {
    if (config_.avail_ram_fn) {
      cached_ram_workers_limit_ =
          ComputeRamWorkersCap(config_.avail_ram_fn(),
                               config_.ram_per_worker_bytes,
                               /*num_gpus=*/1,
                               config_.max_workers_cap);
    }
    active_workers_limit_ =
        std::clamp(std::min(std::max(initial_m, 2), cached_ram_workers_limit_),
                   1,
                   config_.max_workers_cap);
  }
}

ONNXMatchScheduler::WorkerSlot ONNXMatchScheduler::AcquireWorkerSlot() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() { return active_workers_ < active_workers_limit_; });
  ++active_workers_;
  if (config_.adaptive_gpu && !has_vram_baseline_ && config_.avail_vram_fn) {
    size_t vram_free = 0;
    size_t vram_total = 0;
    if (config_.avail_vram_fn(&vram_free, &vram_total) && vram_total > 0) {
      has_vram_baseline_ = true;
      vram_baseline_free_ = vram_free;
      vram_min_free_ = vram_free;
    }
  }
  return WorkerSlot(this);
}

void ONNXMatchScheduler::ReleaseWorkerSlot() {
  bool should_notify = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_workers_;
    should_notify = active_workers_ < active_workers_limit_;
  }
  if (should_notify) {
    cv_.notify_one();
  }
}

void ONNXMatchScheduler::RecordPair(int64_t prep_ns,
                                    int64_t inf_ns,
                                    const int num_runs,
                                    const int pair_max_keypoints) {
  if (!config_.adaptive_gpu || num_runs <= 0) {
    return;
  }

  prep_ns = std::max<int64_t>(0, prep_ns);
  inf_ns = std::max<int64_t>(1, inf_ns);

  bool workers_increased = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pairs_observed_ == 0) {
      ema_prep_ns_ = static_cast<double>(prep_ns);
      ema_inf_ns_ = static_cast<double>(inf_ns);
    } else {
      constexpr double kAlpha = 0.15;
      ema_prep_ns_ =
          kAlpha * static_cast<double>(prep_ns) + (1.0 - kAlpha) * ema_prep_ns_;
      ema_inf_ns_ =
          kAlpha * static_cast<double>(inf_ns) + (1.0 - kAlpha) * ema_inf_ns_;
    }
    ++pairs_observed_;
    ++pairs_at_current_m_;
    max_seen_keypoints_ = std::max(max_seen_keypoints_, pair_max_keypoints);

    int m_cur = inference_limiter_->MaxConcurrent();
    if (config_.fixed_inferences <= 0 &&
        (pairs_at_current_m_ == 4 || pairs_at_current_m_ % 8 == 0) &&
        config_.avail_vram_fn) {
      size_t vram_free = 0;
      size_t vram_total = 0;
      if (config_.avail_vram_fn(&vram_free, &vram_total) && vram_total > 0) {
        if (!has_vram_baseline_) {
          has_vram_baseline_ = true;
          vram_baseline_free_ = vram_free;
          vram_min_free_ = vram_free;
        }
        vram_min_free_ = std::min(vram_min_free_, vram_free);

        const CacheVramUsage cache_vram =
            config_.cache_vram_fn ? config_.cache_vram_fn() : CacheVramUsage{};
        const size_t vram_headroom = std::clamp<size_t>(
            vram_baseline_free_ / 10, 512ULL << 20, 2048ULL << 20);
        const size_t total_drop = vram_baseline_free_ > vram_min_free_
                                      ? (vram_baseline_free_ - vram_min_free_)
                                      : 0;
        const size_t activation_drop =
            total_drop > cache_vram.allocated_bytes
                ? (total_drop - cache_vram.allocated_bytes)
                : 0;

        double kpt_scale = 1.0;
        if (config_.max_num_keypoints > 0 && max_seen_keypoints_ > 0 &&
            max_seen_keypoints_ < config_.max_num_keypoints) {
          kpt_scale = std::min(
              4.0,
              static_cast<double>(config_.max_num_keypoints) /
                  static_cast<double>(std::max(256, max_seen_keypoints_)));
        }
        const int peak_active = inference_limiter_->PeakActiveCount();
        const int materialized_slots =
            std::max(1, std::min(m_cur, peak_active));
        const size_t est_slot_bytes =
            std::max(config_.min_slot_vram_bytes,
                     static_cast<size_t>((static_cast<double>(activation_drop) /
                                          materialized_slots) *
                                         kpt_scale));

        const int worker_bound = config_.fixed_workers > 0
                                     ? config_.fixed_workers
                                     : config_.max_workers_cap;
        const int m_max = std::min({kMaxAutoInferencesPerGpu,
                                    worker_bound,
                                    inference_limiter_->OomCeiling()});

        if (m_cur > 1 && vram_free < vram_headroom / 2) {
          m_cur = std::max(1, m_cur - 1);
          inference_limiter_->SetMaxConcurrent(m_cur);
          pairs_at_current_m_ = 0;
        } else if (m_cur < m_max && peak_active >= m_cur &&
                   pairs_at_current_m_ >= 4) {
          const size_t reserved = vram_headroom + cache_vram.unallocated_bytes;
          const size_t usable_free =
              vram_free > reserved ? (vram_free - reserved) : 0;
          const size_t required = est_slot_bytes + est_slot_bytes / 4;
          if (usable_free >= required) {
            m_cur += 1;
            inference_limiter_->SetMaxConcurrent(m_cur);
            pairs_at_current_m_ = 0;
          }
        }
      }
    }

    if (config_.fixed_workers <= 0) {
      if ((pairs_observed_ == 1 || pairs_observed_ % 16 == 0) &&
          config_.avail_ram_fn) {
        cached_ram_workers_limit_ =
            ComputeRamWorkersCap(config_.avail_ram_fn(),
                                 config_.ram_per_worker_bytes,
                                 /*num_gpus=*/1,
                                 config_.max_workers_cap);
      }
      const double prep_ratio = std::max(0.01, ema_prep_ns_ / ema_inf_ns_);
      const int n_little =
          static_cast<int>(std::ceil(m_cur * (1.0 + prep_ratio)));
      const int target_n =
          std::clamp(std::min(n_little, cached_ram_workers_limit_),
                     std::min(m_cur, cached_ram_workers_limit_),
                     config_.max_workers_cap);
      workers_increased = target_n > active_workers_limit_;
      active_workers_limit_ = target_n;
    }
  }
  if (workers_increased) {
    cv_.notify_all();
  }
}

int ONNXMatchScheduler::ActiveWorkersLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_workers_limit_;
}

int ONNXMatchScheduler::MaxConcurrentInferences() const {
  return inference_limiter_->MaxConcurrent();
}

ONNXFeatureMatcher::ONNXFeatureMatcher(const FeatureMatchingOptions& options)
    : type_(options.type), scheduler_([this, &options]() {
        ONNXMatchScheduler::Config config =
            ONNXMatchScheduler::DefaultConfig(options);
        config.cache_vram_fn = [this]() { return GetCacheVramUsage(); };
        return config;
      }()) {}

void ONNXFeatureMatcher::Match(const Image& image1,
                               const Image& image2,
                               FeatureMatches* matches) {
  THROW_CHECK_NOTNULL(matches);
  matches->clear();

  const auto worker_slot = scheduler_.AcquireWorkerSlot();
  ONNXInferenceLimiter::ConsumeThreadStats();
  const auto start = std::chrono::steady_clock::now();
  MatchImpl(image1, image2, matches);
  const int64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  const auto inf_stats = ONNXInferenceLimiter::ConsumeThreadStats();
  const int64_t prep_ns =
      std::max<int64_t>(0, wall_ns - inf_stats.wait_ns - inf_stats.run_ns);

  int pair_max_keypoints = 0;
  if (image1.keypoints != nullptr) {
    pair_max_keypoints =
        std::max<int>(pair_max_keypoints, image1.keypoints->size());
  } else if (image1.descriptors != nullptr) {
    pair_max_keypoints =
        std::max<int>(pair_max_keypoints, image1.descriptors->data.rows());
  }
  if (image2.keypoints != nullptr) {
    pair_max_keypoints =
        std::max<int>(pair_max_keypoints, image2.keypoints->size());
  } else if (image2.descriptors != nullptr) {
    pair_max_keypoints =
        std::max<int>(pair_max_keypoints, image2.descriptors->data.rows());
  }
  scheduler_.RecordPair(
      prep_ns, inf_stats.run_ns, inf_stats.num_runs, pair_max_keypoints);
}

void ONNXFeatureMatcher::MatchGuided(double max_error,
                                     const Image& image1,
                                     const Image& image2,
                                     TwoViewGeometry* two_view_geometry) {
  LOG(FATAL_THROW) << "Guided matching not supported for "
                   << FeatureMatcherTypeToString(type_) << ".";
}

namespace {

class BruteForceONNXFeatureMatcher : public ONNXFeatureMatcher {
 public:
  explicit BruteForceONNXFeatureMatcher(
      const FeatureMatchingOptions& options,
      const BruteForceONNXMatchingOptions& brute_force_options,
      std::vector<FeatureExtractorType> supported_feature_types,
      bool normalize_descriptors)
      : ONNXFeatureMatcher(options),
        brute_force_options_(brute_force_options),
        supported_feature_types_(std::move(supported_feature_types)),
        normalize_descriptors_(normalize_descriptors),
        model_(brute_force_options.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index,
               inference_limiter()) {
    THROW_CHECK(options.Check());
    THROW_CHECK(!supported_feature_types_.empty());
    THROW_CHECK_EQ(model_.input_shapes().size(), 5);
    ThrowCheckONNXNode(
        model_.input_names()[0], "descs1", model_.input_shapes()[0], {-1, -1});
    ThrowCheckONNXNode(
        model_.input_names()[1], "descs2", model_.input_shapes()[1], {-1, -1});
    ThrowCheckONNXNode(
        model_.input_names()[2], "min_cossim", model_.input_shapes()[2], {});
    ThrowCheckONNXNode(
        model_.input_names()[3], "max_ratio", model_.input_shapes()[3], {});
    ThrowCheckONNXNode(
        model_.input_names()[4], "cross_check", model_.input_shapes()[4], {});
    THROW_CHECK_EQ(model_.output_shapes().size(), 3);
    ThrowCheckONNXNode(
        model_.output_names()[0], "idx0", model_.output_shapes()[0], {-1});
    ThrowCheckONNXNode(
        model_.output_names()[1], "idx1", model_.output_shapes()[1], {-1});
    ThrowCheckONNXNode(
        model_.output_names()[2], "scores", model_.output_shapes()[2], {-1});
  }

 protected:
  void MatchImpl(const Image& image1,
                 const Image& image2,
                 FeatureMatches* matches) override {
    const int num_keypoints1 = image1.descriptors->data.rows();
    const int num_keypoints2 = image2.descriptors->data.rows();
    // Model requires at least 2 descriptors in each set for ratio test.
    if (num_keypoints1 < 2 || num_keypoints2 < 2) {
      return;
    }

    auto create = [this](const Image& image) {
      return FeaturesFromImage(image);
    };
    const std::shared_ptr<const Features> cached1 =
        cache_.GetOrCreate(image1, create);
    const std::shared_ptr<const Features> cached2 =
        cache_.GetOrCreate(image2, create);

    // Create tensors from cached data (tensors must be recreated each call
    // since they reference the underlying data and get consumed by Run()).
    const float min_cossim =
        static_cast<float>(brute_force_options_.min_cossim);
    const float max_ratio = static_cast<float>(brute_force_options_.max_ratio);
    const int64_t cross_check = brute_force_options_.cross_check ? 1 : 0;

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(5);
    input_tensors.emplace_back(CreateONNXTensor(cached1->descriptors_data,
                                                cached1->descriptors_shape));
    input_tensors.emplace_back(CreateONNXTensor(cached2->descriptors_data,
                                                cached2->descriptors_shape));
    input_tensors.emplace_back(CreateONNXScalarTensor(min_cossim));
    input_tensors.emplace_back(CreateONNXScalarTensor(max_ratio));
    input_tensors.emplace_back(CreateONNXScalarTensor(cross_check));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 3);

    // Get num_matches from shape of idx0 output
    const std::vector<int64_t> idx0_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(idx0_shape.size(), 1);
    const int64_t num_matches = idx0_shape[0];

    // Ensure idx1 has the same 1D shape and length as idx0
    const std::vector<int64_t> idx1_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(idx1_shape.size(), 1);
    THROW_CHECK_EQ(idx1_shape[0], num_matches);

    if (num_matches == 0) {
      return;
    }

    const int64_t* idx0_data = output_tensors[0].GetTensorData<int64_t>();
    const int64_t* idx1_data = output_tensors[1].GetTensorData<int64_t>();

    matches->resize(num_matches);
    for (int64_t i = 0; i < num_matches; ++i) {
      FeatureMatch& match = (*matches)[i];
      match.point2D_idx1 = idx0_data[i];
      match.point2D_idx2 = idx1_data[i];
      THROW_CHECK_GE(match.point2D_idx1, 0);
      THROW_CHECK_LT(match.point2D_idx1, num_keypoints1);
      THROW_CHECK_GE(match.point2D_idx2, 0);
      THROW_CHECK_LT(match.point2D_idx2, num_keypoints2);
    }
  }

 private:
  struct Features {
    std::vector<float> descriptors_data;
    std::vector<int64_t> descriptors_shape;
  };

  Features FeaturesFromImage(const Image& image) {
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK(std::find(supported_feature_types_.begin(),
                          supported_feature_types_.end(),
                          image.descriptors->type) !=
                supported_feature_types_.end())
        << "Unsupported feature type: "
        << FeatureExtractorTypeToString(image.descriptors->type);
    FeatureDescriptorsFloat descriptors = image.descriptors->ToFloat();
    if (normalize_descriptors_) {
      L2NormalizeFeatureDescriptors(&descriptors.data);
    }

    const int num_keypoints = descriptors.data.rows();
    const int descriptor_dim = descriptors.data.cols();
    THROW_CHECK_GT(descriptor_dim, 0);

    Features features;
    features.descriptors_shape = {num_keypoints, descriptor_dim};
    features.descriptors_data.assign(
        descriptors.data.data(),
        descriptors.data.data() + descriptors.data.size());

    return features;
  }

  const BruteForceONNXMatchingOptions brute_force_options_;
  const std::vector<FeatureExtractorType> supported_feature_types_;
  const bool normalize_descriptors_;
  ONNXModel model_;

  // Cached features for avoiding redundant data copies across workers.
  ThreadSafeImageFeatureCache<Features> cache_;
};

}  // namespace

#endif  // COLMAP_ONNX_ENABLED

bool BruteForceONNXMatchingOptions::Check() const {
  CHECK_OPTION_IN(min_cossim, -1, 1);
  CHECK_OPTION_IN(max_ratio, 0, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateBruteForceONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const BruteForceONNXMatchingOptions& brute_force_options,
    std::vector<FeatureExtractorType> supported_feature_types,
    bool normalize_descriptors) {
#ifdef COLMAP_ONNX_ENABLED
  THROW_CHECK(brute_force_options.Check());
  return std::make_unique<BruteForceONNXFeatureMatcher>(
      options,
      brute_force_options,
      std::move(supported_feature_types),
      normalize_descriptors);
#else
  throw std::runtime_error("ONNX brute-force matching requires ONNX support.");
#endif
}

}  // namespace colmap
