#pragma once

#include <colmap/util/logging.h>

#include <iostream>
#include <memory>

#include <boost/program_options.hpp>

namespace glomap {

struct GlobalMapperOptions;
struct ViewGraphCalibratorOptions;
struct RelativePoseEstimationOptions;
struct RotationEstimatorOptions;
struct TrackEstablishmentOptions;
struct GlobalPositionerOptions;
struct BundleAdjusterOptions;
struct TriangulatorOptions;
struct InlierThresholdOptions;
struct GravityRefinerOptions;

class OptionManager {
 public:
  explicit OptionManager(bool add_project_options = true);
  void AddAllOptions();
  void AddDatabaseOptions();
  void AddImageOptions();
  void AddGlobalMapperOptions();
  void AddGlobalMapperFullOptions();
  void AddGlobalMapperResumeOptions();
  void AddGlobalMapperResumeFullOptions();
  void AddViewGraphCalibrationOptions();
  void AddRelativePoseEstimationOptions();
  void AddRotationEstimatorOptions();
  void AddTrackEstablishmentOptions();
  void AddGlobalPositionerOptions();
  void AddBundleAdjusterOptions();
  void AddTriangulatorOptions();
  void AddInlierThresholdOptions();
  void AddGravityRefinerOptions();

  template <typename T>
  void AddRequiredOption(const std::string& name,
                         T* option,
                         const std::string& help_text = "");
  template <typename T>
  void AddDefaultOption(const std::string& name,
                        T* option,
                        const std::string& help_text = "");

  void Reset();
  void ResetOptions(bool reset_paths);

  void Parse(int argc, char** argv);

  std::shared_ptr<std::string> database_path;
  std::shared_ptr<std::string> image_path;

  std::shared_ptr<GlobalMapperOptions> mapper;
  std::shared_ptr<GravityRefinerOptions> gravity_refiner;

 private:
  template <typename T>
  void AddAndRegisterRequiredOption(const std::string& name,
                                    T* option,
                                    const std::string& help_text = "");
  template <typename T>
  void AddAndRegisterDefaultOption(const std::string& name,
                                   T* option,
                                   const std::string& help_text = "");

  template <typename T>
  void RegisterOption(const std::string& name, const T* option);

  std::shared_ptr<boost::program_options::options_description> desc_;

  std::vector<std::pair<std::string, const bool*>> options_bool_;
  std::vector<std::pair<std::string, const int*>> options_int_;
  std::vector<std::pair<std::string, const double*>> options_double_;
  std::vector<std::pair<std::string, const std::string*>> options_string_;

  bool added_database_options_ = false;
  bool added_image_options_ = false;
  bool added_mapper_options_ = false;
  bool added_view_graph_calibration_options_ = false;
  bool added_relative_pose_options_ = false;
  bool added_rotation_averaging_options_ = false;
  bool added_track_establishment_options_ = false;
  bool added_global_positioning_options_ = false;
  bool added_bundle_adjustment_options_ = false;
  bool added_triangulation_options_ = false;
  bool added_inliers_options_ = false;
  bool added_gravity_refiner_options_ = false;
};

template <typename T>
void OptionManager::AddRequiredOption(const std::string& name,
                                      T* option,
                                      const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
}

template <typename T>
void OptionManager::AddDefaultOption(const std::string& name,
                                     T* option,
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
    LOG(ERROR) << "Unsupported option type: " << name;
  }
}

}  // namespace glomap
