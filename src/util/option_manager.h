// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_UTIL_OPTION_MANAGER_H_
#define COLMAP_SRC_UTIL_OPTION_MANAGER_H_

#include <memory>

#include <boost/program_options.hpp>

#include "base/feature_extraction.h"
#include "base/feature_matching.h"
#include "controllers/incremental_mapper.h"
#include "mvs/fusion.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "optim/bundle_adjustment.h"
#include "ui/render_options.h"

namespace colmap {

class OptionManager {
 public:
  OptionManager();

  // Create "optimal" set of options for different reconstruction scenarios.
  void InitForIndividualData();
  void InitForVideoData();
  void InitForInternetData();

  void AddAllOptions();
  void AddLogOptions();
  void AddDatabaseOptions();
  void AddImageOptions();
  void AddExtractionOptions();
  void AddMatchingOptions();
  void AddExhaustiveMatchingOptions();
  void AddSequentialMatchingOptions();
  void AddVocabTreeMatchingOptions();
  void AddSpatialMatchingOptions();
  void AddTransitiveMatchingOptions();
  void AddBundleAdjustmentOptions();
  void AddMapperOptions();
  void AddDenseStereoOptions();
  void AddDenseFusionOptions();
  void AddDenseMeshingOptions();
  void AddRenderOptions();

  template <typename T>
  void AddRequiredOption(const std::string& name, T* option,
                         const std::string& help_text = "");
  template <typename T>
  void AddDefaultOption(const std::string& name, T* option,
                        const std::string& help_text = "");

  void Reset();
  bool Check();

  void Parse(const int argc, char** argv);
  bool Read(const std::string& path);
  bool ReRead(const std::string& path);
  void Write(const std::string& path) const;

  std::shared_ptr<std::string> project_path;
  std::shared_ptr<std::string> database_path;
  std::shared_ptr<std::string> image_path;

  std::shared_ptr<ImageReader::Options> image_reader;
  std::shared_ptr<SiftExtractionOptions> sift_extraction;
  std::shared_ptr<SiftCPUFeatureExtractor::Options> sift_cpu_extraction;
  std::shared_ptr<SiftGPUFeatureExtractor::Options> sift_gpu_extraction;

  std::shared_ptr<SiftMatchingOptions> sift_matching;
  std::shared_ptr<ExhaustiveFeatureMatcher::Options> exhaustive_matching;
  std::shared_ptr<SequentialFeatureMatcher::Options> sequential_matching;
  std::shared_ptr<VocabTreeFeatureMatcher::Options> vocab_tree_matching;
  std::shared_ptr<SpatialFeatureMatcher::Options> spatial_matching;
  std::shared_ptr<TransitiveFeatureMatcher::Options> transitive_matching;

  std::shared_ptr<BundleAdjuster::Options> bundle_adjustment;
  std::shared_ptr<IncrementalMapperController::Options> mapper;

  std::shared_ptr<mvs::PatchMatch::Options> dense_stereo;
  std::shared_ptr<mvs::StereoFusion::Options> dense_fusion;
  std::shared_ptr<mvs::PoissonReconstructionOptions> dense_meshing;

  std::shared_ptr<RenderOptions> render;

 private:
  template <typename T>
  void AddAndRegisterRequiredOption(const std::string& name, T* option,
                                    const std::string& help_text = "");
  template <typename T>
  void AddAndRegisterDefaultOption(const std::string& name, T* option,
                                   const std::string& help_text = "");

  template <typename T>
  void RegisterOption(const std::string& name, const T* option);

  std::shared_ptr<boost::program_options::options_description> desc_;

  std::vector<std::pair<std::string, const bool*>> options_bool_;
  std::vector<std::pair<std::string, const int*>> options_int_;
  std::vector<std::pair<std::string, const double*>> options_double_;
  std::vector<std::pair<std::string, const std::string*>> options_string_;

  bool added_log_options_;
  bool added_database_options_;
  bool added_image_options_;
  bool added_extraction_options_;
  bool added_match_options_;
  bool added_exhaustive_match_options_;
  bool added_sequential_match_options_;
  bool added_vocab_tree_match_options_;
  bool added_spatial_match_options_;
  bool added_transitive_match_options_;
  bool added_ba_options_;
  bool added_mapper_options_;
  bool added_dense_stereo_options_;
  bool added_dense_fusion_options_;
  bool added_dense_meshing_options_;
  bool added_render_options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void OptionManager::AddRequiredOption(const std::string& name, T* option,
                                      const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
}

template <typename T>
void OptionManager::AddDefaultOption(const std::string& name, T* option,
                                     const std::string& help_text) {
  desc_->add_options()(
      name.c_str(),
      boost::program_options::value<T>(option)->default_value(*option),
      help_text.c_str());
}

template <typename T>
void OptionManager::AddAndRegisterRequiredOption(const std::string& name,
                                                 T* option,
                                                 const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
  RegisterOption(name, option);
}

template <typename T>
void OptionManager::AddAndRegisterDefaultOption(const std::string& name,
                                                T* option,
                                                const std::string& help_text) {
  desc_->add_options()(
      name.c_str(),
      boost::program_options::value<T>(option)->default_value(*option),
      help_text.c_str());
  RegisterOption(name, option);
}

template <typename T>
void OptionManager::RegisterOption(const std::string& name, const T* option) {
  if (std::is_same<T, bool>::value) {
    options_bool_.emplace_back(name, reinterpret_cast<const bool*>(option));
  } else if (std::is_same<T, int>::value) {
    options_int_.emplace_back(name, reinterpret_cast<const int*>(option));
  } else if (std::is_same<T, double>::value) {
    options_double_.emplace_back(name, reinterpret_cast<const double*>(option));
  } else if (std::is_same<T, std::string>::value) {
    options_string_.emplace_back(name,
                                 reinterpret_cast<const std::string*>(option));
  } else {
    LOG(FATAL) << "Unsupported option type";
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_OPTION_MANAGER_H_
