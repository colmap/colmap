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

#include <filesystem>
#include <fstream>

#include <boost/property_tree/ini_parser.hpp>

namespace config = boost::program_options;

namespace colmap {

BaseOptionManager::BaseOptionManager(bool add_project_options) {
  project_path = std::make_shared<std::string>();
  database_path = std::make_shared<std::string>();
  image_path = std::make_shared<std::string>();

  ResetImpl();

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

  AddAndRegisterDefaultOption("default_random_seed", &kDefaultPRNGSeed);
}

void BaseOptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  AddAndRegisterDefaultOption("log_to_stderr", &FLAGS_logtostderr);
  AddAndRegisterDefaultOption("log_level", &FLAGS_v);
}

void BaseOptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddAndRegisterRequiredOption("database_path", database_path.get());
}

void BaseOptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddAndRegisterRequiredOption("image_path", image_path.get());
}

void BaseOptionManager::Reset() { ResetImpl(); }

void BaseOptionManager::ResetOptions(const bool reset_paths) {
  ResetOptionsImpl(reset_paths);
}

void BaseOptionManager::ResetImpl() {
  FLAGS_logtostderr = true;

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
              CHECK_OPTION_IMPL(database_parent_path == "" ||
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

bool BaseOptionManager::Read(const std::string& path) {
  config::variables_map vmap;

  if (!ExistsFile(path)) {
    LOG(ERROR) << "Configuration file does not exist.";
    return false;
  }

  try {
    std::ifstream file(path);
    THROW_CHECK_FILE_OPEN(file, path);
    config::store(config::parse_config_file(file, *desc_), vmap);
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

bool BaseOptionManager::ReRead(const std::string& path) {
  Reset();
  AddAllOptions();
  return Read(path);
}

void BaseOptionManager::Write(const std::string& path) const {
  boost::property_tree::ptree pt;

  // First, put all options without a section and then those with a section.
  // This is necessary as otherwise older Boost versions will write the
  // options without a section in between other sections and therefore
  // the errors will be assigned to the wrong section if read later.

  for (const auto& option : options_bool_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_path_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, option.second->string());
    }
  }

  for (const auto& option : options_bool_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_path_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, option.second->string());
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
