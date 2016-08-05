// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_UTIL_MISC_H_
#define COLMAP_SRC_UTIL_MISC_H_

#include <iostream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace colmap {

// Append trailing slash to string if it does not yet end with a slash.
//
// @param str    Input string.
//
// @return       Input string, with a trailing slash.
std::string EnsureTrailingSlash(const std::string& str);

// Check whether file name has the file extension (case insensitive).
bool HasFileExtension(const std::string& file_name, const std::string& ext);

// Return list of files, recursively in all sub-directories.
std::vector<std::string> GetRecursiveFileList(const std::string& path);

// Print first-order heading with over- and underscores to `std::cout`.
//
// @param heading      Heading text as a single line.
void PrintHeading1(const std::string& heading);

// Print second-order heading with underscores to `std::cout`.
//
// @param heading      Heading text as a single line.
void PrintHeading2(const std::string& heading);

// Format string by replacing embedded format specifiers with their respective
// values, see `printf` for more details. This is a modified implementation
// of Google's BSD-licensed StringPrintf function.
std::string StringPrintf(const char* format, ...);

// Replace all occurrences of `old_str` with `new_str` in the given string.
//
// @param str      String to which to apply the replacement.
// @param old_str  Old string token for replacement.
// @param new_str  New string token for replacement.
//
// @ return        String with all occurences of old replaced with new string.
std::string StringReplace(const std::string& str, const std::string& old_str,
                          const std::string& new_str);

// Split string into list of words using the given delimiters.
//
// @param str    String to split.
// @param delim  Delimiters used to split the string. May contain multiple
//               delimiters in the same string.
//
// @return       The words of the string.
std::vector<std::string> StringSplit(const std::string& str,
                                     const std::string& delim);

// Check whether a string starts with a certain prefix.
bool StringStartsWith(const std::string& str, const std::string& prefix);

// Check if vector contains elements.
template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, const T value);

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector);

// Parse CSV line to a list of values.
//
// @param values    The comma-separated string as a single line.
//
// @return   T       he elements of the CSV line.
template <typename T>
std::vector<T> CSVToVector(const std::string& csv);

// Concatenate values in list to comma-separated list.
//
// @param values     The list of elements to concatenate.
//
// @return           The concatenated list of elements as string.
template <typename T>
std::string VectorToCSV(const std::vector<T>& values);

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
std::vector<T> CSVToVector(const std::string& csv) {
  auto elems = StringSplit(csv, ",;");
  std::vector<T> values;
  values.reserve(elems.size());
  for (auto& elem : elems) {
    boost::erase_all(elem, " ");
    if (elem.empty()) {
      continue;
    }
    try {
      values.push_back(boost::lexical_cast<T>(elem));
    } catch (std::exception) {
      return std::vector<T>(0);
    }
  }
  return values;
}

template <typename T>
std::string VectorToCSV(const std::vector<T>& values) {
  std::string string;
  for (const T value : values) {
    string += std::to_string(value) + ", ";
  }
  return string.substr(0, string.length() - 2);
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_MISC_H_
