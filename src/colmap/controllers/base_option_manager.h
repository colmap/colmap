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

#pragma once

#include "colmap/util/logging.h"

#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

namespace colmap {

// Base class for option managers providing core infrastructure for
// command-line parsing, configuration file I/O, and option registration.
class BaseOptionManager {
 public:
  explicit BaseOptionManager(bool add_project_options = true);
  virtual ~BaseOptionManager() = default;

  void AddRandomOptions();
  void AddLogOptions();
  void AddDatabaseOptions();
  void AddImageOptions();

  template <typename T>
  void AddRequiredOption(const std::string& name,
                         T* option,
                         const std::string& help_text = "");
  template <typename T>
  void AddDefaultOption(const std::string& name,
                        T* option,
                        const std::string& help_text = "");

  virtual void Reset();
  virtual void ResetOptions(bool reset_paths);

  virtual bool Check();

  bool Parse(int argc, char** argv);
  virtual bool Read(const std::string& path);
  bool ReRead(const std::string& path);
  void Write(const std::string& path) const;

  std::shared_ptr<std::string> project_path;
  std::shared_ptr<std::string> database_path;
  std::shared_ptr<std::string> image_path;

 protected:
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

  // Hook for subclasses to perform post-parse processing.
  // Called after successful parsing but before Check().
  virtual void PostParse();

  // Hook for subclasses to print custom help message.
  virtual void PrintHelp() const;

  // Hook for subclasses to add all their options. Called by ReRead().
  // Base implementation adds common options (random, log, database, image).
  // Subclasses should call BaseOptionManager::AddAllOptions() first.
  virtual void AddAllOptions();

  std::shared_ptr<boost::program_options::options_description> desc_;

  std::vector<std::pair<std::string, const bool*>> options_bool_;
  std::vector<std::pair<std::string, const int*>> options_int_;
  std::vector<std::pair<std::string, const double*>> options_double_;
  std::vector<std::pair<std::string, const std::string*>> options_string_;

  bool added_random_options_ = false;
  bool added_log_options_ = false;
  bool added_database_options_ = false;
  bool added_image_options_ = false;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void BaseOptionManager::AddRequiredOption(const std::string& name,
                                          T* option,
                                          const std::string& help_text) {
  desc_->add_options()(name.c_str(),
                       boost::program_options::value<T>(option)->required(),
                       help_text.c_str());
}

template <typename T>
void BaseOptionManager::AddDefaultOption(const std::string& name,
                                         T* option,
                                         const std::string& help_text) {
  if constexpr (std::is_floating_point<T>::value) {
    desc_->add_options()(
        name.c_str(),
        boost::program_options::value<T>(option)->default_value(
            *option, StringPrintf("%.3g", *option)),
        help_text.c_str());
  } else {
    desc_->add_options()(
        name.c_str(),
        boost::program_options::value<T>(option)->default_value(*option),
        help_text.c_str());
  }
}

template <typename T>
void BaseOptionManager::AddAndRegisterRequiredOption(
    const std::string& name, T* option, const std::string& help_text) {
  AddRequiredOption(name, option, help_text);
  RegisterOption(name, option);
}

template <typename T>
void BaseOptionManager::AddAndRegisterDefaultOption(
    const std::string& name, T* option, const std::string& help_text) {
  AddDefaultOption(name, option, help_text);
  RegisterOption(name, option);
}

template <typename T>
void BaseOptionManager::RegisterOption(const std::string& name,
                                       const T* option) {
  if constexpr (std::is_same<T, bool>::value) {
    options_bool_.emplace_back(name, reinterpret_cast<const bool*>(option));
  } else if constexpr (std::is_same<T, int>::value) {
    options_int_.emplace_back(name, reinterpret_cast<const int*>(option));
  } else if constexpr (std::is_same<T, double>::value) {
    options_double_.emplace_back(name, reinterpret_cast<const double*>(option));
  } else if constexpr (std::is_same<T, std::string>::value) {
    options_string_.emplace_back(name,
                                 reinterpret_cast<const std::string*>(option));
  } else {
    LOG(FATAL_THROW) << "Unsupported option type";
  }
}

}  // namespace colmap
