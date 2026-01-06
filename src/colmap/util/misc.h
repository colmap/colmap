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
#include "colmap/util/string.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace colmap {

#ifndef STRINGIFY
#define STRINGIFY(s) STRINGIFY_(s)
#define STRINGIFY_(s) #s
#endif  // STRINGIFY

// Log first-order heading with over- and underscores.
void PrintHeading1(const std::string& heading);

// Log second-order heading with underscores.
void PrintHeading2(const std::string& heading);

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
        values.push_back(std::stod(elem));
      } else if constexpr (std::is_same<T, double>::value) {
        values.push_back(std::stold(elem));
      }
    } catch (const std::invalid_argument&) {
      LOG(ERROR) << "Failed to convert CSV element: " << elem;
      return {};
    }
  }
  return values;
}

}  // namespace colmap
