// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/mat.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/util/enum_utils.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

class MVSEstimator {
 public:
  MAKE_ENUM_CLASS(Type, 0, PATCH_MATCH, MVSFORMER_PP);
  enum class Pass { PHOTOMETRIC, GEOMETRIC };

  struct Problem {
    int ref_image_idx = -1;
    std::vector<int> src_image_idxs;
    std::vector<Image>* images = nullptr;
    std::vector<DepthMap>* depth_maps = nullptr;
    std::vector<NormalMap>* normal_maps = nullptr;
    std::vector<Mat<float>>* confidence_maps = nullptr;
    float depth_min = -1.0f;
    float depth_max = -1.0f;

    void Print() const;
  };

  struct Result {
    DepthMap depth_map;
    NormalMap normal_map;
    std::optional<Mat<float>> confidence_map;
    std::optional<ConsistencyGraph> consistency_graph;
  };

  struct Capabilities {
    bool requires_rgb = false;
    bool supports_geometric_pass = true;
    bool supports_repeated_source_images = false;
    size_t min_num_source_images = 1;
    std::optional<size_t> max_num_source_images;
  };

  struct Options;

  virtual ~MVSEstimator() = default;

  static std::unique_ptr<MVSEstimator> Create(const Options& options,
                                              int device_index);

  virtual Capabilities GetCapabilities() const = 0;
  virtual Result Estimate(const Problem& problem, Pass pass) = 0;
};

class PatchMatchStereo : public MVSEstimator {
 public:
  struct Options {
    double depth_min = -1.0f;
    double depth_max = -1.0f;
    double sigma_spatial = -1;
    double sigma_color = 0.2f;
    double ncc_sigma = 0.6f;
    double min_triangulation_angle = 1.0f;
    double incident_angle_sigma = 0.9f;
    double geom_consistency_regularizer = 0.3f;
    double geom_consistency_max_cost = 3.0f;
    double filter_min_ncc = 0.1f;
    double filter_min_triangulation_angle = 3.0f;
    double filter_geom_consistency_max_cost = 1.0f;
    double cache_size = 32.0;
    std::string gpu_index = "-1";
    int max_image_size = -1;
    int window_radius = 5;
    int window_step = 1;
    int num_samples = 15;
    int num_iterations = 5;
    int filter_min_num_consistent = 2;
    int num_threads = -1;
    bool geom_consistency = true;
    bool filter = true;
    bool allow_missing_files = false;
    bool write_consistency_graph = false;

    void Print() const;
    bool Check() const;
  };

  explicit PatchMatchStereo(Options options);
  Capabilities GetCapabilities() const override;
  Result Estimate(const Problem& problem, Pass pass) override;

 private:
  Options options_;
};

class MVSFormerPlusPlus : public MVSEstimator {
 public:
  struct Options {
    std::string model_path;
    int num_views = 5;
    int max_image_size = 1536;
    double depth_min = -1.0;
    double depth_max = -1.0;
    double min_confidence = 0.5;
    double filter_max_reproj_error = 1.0;
    double filter_max_depth_error = 0.01;
    double filter_max_normal_error = 10.0;
    double cache_size = 32.0;
    std::string gpu_index = "-1";
    int filter_min_num_consistent = 2;
    int num_threads = -1;
    bool use_gpu = true;
    bool geom_consistency = true;
    bool allow_missing_files = false;
    bool write_consistency_graph = false;

    void Print() const;
    bool Check() const;
  };

  MVSFormerPlusPlus(Options options, int device_index);
  ~MVSFormerPlusPlus() override;

  Capabilities GetCapabilities() const override;
  Result Estimate(const Problem& problem, Pass pass) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct MVSEstimator::Options {
  explicit Options(Type type = DefaultType());

  static Type DefaultType();

  std::shared_ptr<PatchMatchStereo::Options> patch_match;
  std::shared_ptr<MVSFormerPlusPlus::Options> mvsformer_pp;
  Type type;

  Options(const Options& other);
  Options& operator=(const Options& other);
  Options(Options&& other) = default;
  Options& operator=(Options&& other) = default;

  bool Check() const;
};

using PatchMatchOptions = PatchMatchStereo::Options;

}  // namespace mvs
}  // namespace colmap
