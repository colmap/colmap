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

#include "colmap/controllers/base_option_manager.h"

#include "colmap/math/random.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <fstream>

#include <boost/property_tree/ini_parser.hpp>

namespace config = boost::program_options;

namespace colmap {

BaseOptionManager::BaseOptionManager(bool add_project_options) {
  project_path = std::make_shared<std::filesystem::path>();
  database_path = std::make_shared<std::filesystem::path>();
  image_path = std::make_shared<std::filesystem::path>();

  ResetImpl(/*reset_logging=*/true);

  desc_->add_options()("help,h", "");
  if (add_project_options) {
    desc_->add_options()("project_path", config::value<std::string>());
  }

  AddRandomOptions();
  AddLogOptions();
}

void BaseOptionManager::AddRandomOptions() {
  if (added_random_options_) {
    return;
  }
  added_random_options_ = true;

  AddDefaultOption("default_random_seed", &kDefaultPRNGSeed);
}

void BaseOptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  AddDefaultOption(
      "log_target", &log_target_, "{stderr, stdout, file, stderr_and_file}");
  // Directory for log files. If empty, glog uses $GOOGLE_LOG_DIR, /tmp, or
  // %TEMP%.
  AddDefaultOption("log_path", &FLAGS_log_dir);
  AddDefaultOption("log_level", &FLAGS_v);
  AddDefaultOption("log_severity",
                   &FLAGS_minloglevel,
                   "0:INFO, 1:WARNING, 2:ERROR, 3:FATAL");
  AddDefaultOption("log_color", &FLAGS_colorlogtostderr);
}

void BaseOptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddRequiredOption("database_path", database_path.get());
}

void BaseOptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddRequiredOption("image_path", image_path.get());
}

void BaseOptionManager::Reset(bool reset_logging) { ResetImpl(reset_logging); }

void BaseOptionManager::ResetOptions(const bool reset_paths) {
  ResetOptionsImpl(reset_paths);
}

void BaseOptionManager::ResetImpl(bool reset_logging) {
  if (reset_logging) {
    log_target_ = "stderr_and_file";
    FLAGS_log_dir = "";
    FLAGS_v = 0;
    FLAGS_minloglevel = 0;
    FLAGS_colorlogtostderr = true;
    ApplyLogFlags();
  }

  const bool kResetPaths = true;
  ResetOptionsImpl(kResetPaths);

  desc_ = std::make_shared<boost::program_options::options_description>();

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();
  options_path_.clear();

  added_random_options_ = false;
  added_log_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
}

void BaseOptionManager::ResetOptionsImpl(const bool reset_paths) {
  if (reset_paths) {
    *project_path = "";
    *database_path = "";
    *image_path = "";
  }
}

bool BaseOptionManager::Check() {
  bool success = true;

  if (added_database_options_) {
    const auto database_parent_path = GetParentDir(*database_path);
    success = success && CHECK_OPTION_IMPL(!ExistsDir(*database_path)) &&
              CHECK_OPTION_IMPL(database_parent_path.empty() ||
                                ExistsDir(database_parent_path));
  }

  if (added_image_options_) {
    success = success && CHECK_OPTION_IMPL(ExistsDir(*image_path));
  }

  return success;
}

void BaseOptionManager::PostParse() {
  // Default implementation does nothing. Subclasses can override.
}

void BaseOptionManager::ApplyEnumConversions() {
  for (const auto& info : enum_options_) {
    info->apply();
  }
}

void BaseOptionManager::ApplyLogFlags() {
  FLAGS_logtostderr = false;
#if defined(GLOG_VERSION_MAJOR) && \
    (GLOG_VERSION_MAJOR > 0 || GLOG_VERSION_MINOR >= 6)
  FLAGS_logtostdout = false;
#endif
  FLAGS_alsologtostderr = false;

  if (log_target_ == "stderr") {
    FLAGS_logtostderr = true;
  } else if (log_target_ == "stdout") {
#if defined(GLOG_VERSION_MAJOR) && \
    (GLOG_VERSION_MAJOR > 0 || GLOG_VERSION_MINOR >= 6)
    FLAGS_logtostdout = true;
#else
    LOG(WARNING) << "log_target=stdout requires glog >= 0.6. "
                    "Falling back to stderr.";
    FLAGS_logtostderr = true;
#endif
  } else if (log_target_ == "file") {
  } else if (log_target_ == "stderr_and_file") {
    FLAGS_alsologtostderr = true;
  } else {
    LOG(ERROR) << "Invalid log_target: " << log_target_
               << ". Falling back to stderr_and_file.";
    FLAGS_alsologtostderr = true;
  }

#if defined(GLOG_VERSION_MAJOR) && \
    (GLOG_VERSION_MAJOR > 0 || GLOG_VERSION_MINOR >= 6)
  FLAGS_colorlogtostdout = FLAGS_colorlogtostderr;
#endif

  if (!FLAGS_log_dir.empty() &&
      (log_target_ == "file" || log_target_ == "stderr_and_file")) {
    CreateDirIfNotExists(FLAGS_log_dir);
  }
}

void BaseOptionManager::PrintHelp() const {
  LOG(INFO) << "Options can either be specified via command-line or by "
               "defining them in a .ini project file.\n"
            << *desc_;
}

void BaseOptionManager::AddAllOptions() {
  AddRandomOptions();
  AddLogOptions();
  AddDatabaseOptions();
  AddImageOptions();
}

bool BaseOptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      PrintHelp();
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      exit(EXIT_SUCCESS);
    }

    if (vmap.count("project_path")) {
      *project_path = vmap["project_path"].as<std::string>();
      if (!Read(*project_path)) {
        return false;
      }
    } else {
      vmap.notify();
    }

    ApplyEnumConversions();
    ApplyLogFlags();
    PostParse();

  } catch (std::exception& exc) {
    LOG(ERROR) << "Failed to parse options - " << exc.what() << ".";
    return false;
  } catch (...) {
    LOG(ERROR) << "Failed to parse options for unknown reason.";
    return false;
  }

  if (!Check()) {
    LOG(ERROR) << "Invalid options provided.";
    return false;
  }

  return true;
}

bool BaseOptionManager::Read(const std::filesystem::path& path,
                             bool allow_unregistered) {
  config::variables_map vmap;

  if (!ExistsFile(path)) {
    LOG(ERROR) << "Configuration file does not exist.";
    return false;
  }

  try {
    std::ifstream file(path);
    THROW_CHECK_FILE_OPEN(file, path);

    const config::parsed_options parsed_options =
        config::parse_config_file(file, *desc_, allow_unregistered);
    config::store(parsed_options, vmap);

    if (allow_unregistered) {
      for (const auto& option : parsed_options.options) {
        if (option.unregistered) {
          LOG(WARNING) << "Unrecognized option key: " << option.string_key;
        }
      }
    }

    vmap.notify();
    ApplyEnumConversions();
  } catch (std::exception& e) {
    LOG(ERROR) << "Failed to parse options " << e.what() << ".";
    return false;
  } catch (...) {
    LOG(ERROR) << "Failed to parse options for unknown reason.";
    return false;
  }

  return true;
}

bool BaseOptionManager::ReRead(const std::filesystem::path& path,
                               bool reset_logging,
                               bool allow_unregistered) {
  Reset(reset_logging);
  AddAllOptions();
  return Read(path, allow_unregistered);
}

void BaseOptionManager::Write(const std::filesystem::path& path) const {
  boost::property_tree::ptree pt;

  // First, put all options without a section and then those with a section.
  // This is necessary as otherwise older Boost versions will write the
  // options without a section in between other sections and therefore
  // the errors will be assigned to the wrong section if read later.

  for (const auto& [key, value] : options_bool_) {
    if (!StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_int_) {
    if (!StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_double_) {
    if (!StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_string_) {
    if (!StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_path_) {
    if (!StringContains(key, ".")) {
      pt.put(key, value->string());
    }
  }

  for (const auto& [key, value] : options_bool_) {
    if (StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_int_) {
    if (StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_double_) {
    if (StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_string_) {
    if (StringContains(key, ".")) {
      pt.put(key, *value);
    }
  }

  for (const auto& [key, value] : options_path_) {
    if (StringContains(key, ".")) {
      pt.put(key, value->string());
    }
  }

  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);
  boost::property_tree::write_ini(file, pt);
  file.close();
}

}  // namespace colmap
