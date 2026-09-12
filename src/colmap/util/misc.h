// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace colmap {

#ifndef STRINGIFY
#define STRINGIFY(s) STRINGIFY_(s)
#define STRINGIFY_(s) #s
#endif  // STRINGIFY

#ifndef LOG_HEADING1
#define LOG_HEADING1(message) LOG(INFO) << "=== " << (message) << " ===";
#endif  // LOG_HEADING1

#ifndef LOG_HEADING2
#define LOG_HEADING2(message) LOG(INFO) << "--- " << (message) << " ---";
#endif  // LOG_HEADING2

// Check if vector contains elements.
template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, T value);

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector);

// Parse CSV line to a list of values.
template <typename T>
std::vector<T> CSVToVector(const std::string& csv);

// Concatenate values in list to comma-separated list.
template <typename T>
std::string VectorToCSV(const std::vector<T>& values);

// Remove an argument from the list of command-line arguments.
void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, const T value) {
  return std::find_if(vector.begin(), vector.end(), [value](const T element) {
           return element == value;
         }) != vector.end();
}

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector) {
  std::vector<T> unique_vector = vector;
  return std::unique(unique_vector.begin(), unique_vector.end()) !=
         unique_vector.end();
}

template <typename T>
std::string VectorToCSV(const std::vector<T>& values) {
  if (values.empty()) {
    return "";
  }

  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  for (const T& value : values) {
    stream << value << ", ";
  }
  std::string buf = stream.str();
  buf.resize(buf.size() - 2);
  return buf;
}

template <typename T>
std::vector<T> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<T> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    StringTrim(&elem);
    if (elem.empty()) {
      continue;
    }
    try {
      static_assert(std::is_same<T, std::string>::value ||  //
                        std::is_same<T, int>::value ||      //
                        std::is_same<T, float>::value ||    //
                        std::is_same<T, double>::value,     //
                    "Unsupported type");
      if constexpr (std::is_same<T, std::string>::value) {
        values.push_back(std::move(elem));
      } else if constexpr (std::is_same<T, int>::value) {
        values.push_back(std::stoi(elem));
      } else if constexpr (std::is_same<T, float>::value) {
        values.push_back(static_cast<float>(StringToDouble(elem)));
      } else if constexpr (std::is_same<T, double>::value) {
        values.push_back(StringToDouble(elem));
      }
    } catch (const std::invalid_argument&) {
      LOG(ERROR) << "Failed to convert CSV element: " << elem;
      return {};
    }
  }
  return values;
}

}  // namespace colmap
