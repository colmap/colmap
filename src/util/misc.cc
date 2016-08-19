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

#include "util/misc.h"

#include <cstdarg>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>

#include <boost/algorithm/string.hpp>

namespace colmap {

std::string EnsureTrailingSlash(const std::string& str) {
  if (str.length() > 0) {
    if (str.back() != '/') {
      return str + "/";
    }
  } else {
    return str + "/";
  }
  return str;
}

bool HasFileExtension(const std::string& file_name, const std::string& ext) {
  CHECK(!ext.empty());
  CHECK_EQ(ext.at(0), '.');
  std::string ext_lower = ext;
  boost::to_lower(ext_lower);
  if (file_name.size() >= ext_lower.size() &&
      file_name.substr(file_name.size() - ext_lower.size(), ext_lower.size()) ==
          ext_lower) {
    return true;
  }
  return false;
}

void CreateDirIfNotExists(const std::string& path) {
  if (!boost::filesystem::is_directory(path)) {
    CHECK(boost::filesystem::create_directory(path));
  }
}

std::string GetPathBaseName(const std::string& path) {
  const std::vector<std::string> names =
      StringSplit(StringReplace(path, "\\", "/"), "/");
  if (names.size() > 1 && names.back() == "") {
    return names[names.size() - 2];
  } else {
    return names.back();
  }
}

std::vector<std::string> GetRecursiveFileList(const std::string& path) {
  std::vector<std::string> file_list;
  for (auto it = boost::filesystem::recursive_directory_iterator(path);
       it != boost::filesystem::recursive_directory_iterator(); ++it) {
    if (boost::filesystem::is_regular_file(*it)) {
      const boost::filesystem::path file_path = *it;
      file_list.push_back(file_path.string());
    }
  }
  return file_list;
}

void PrintHeading1(const std::string& heading) {
  std::cout << std::endl << std::string(78, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(78, '=') << std::endl << std::endl;
}

void PrintHeading2(const std::string& heading) {
  std::cout << std::endl << heading << std::endl;
  std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

bool IsBigEndian() {
  union {
    uint32_t i;
    char c[4];
  } bin = {0x01020304};
  return bin.c[0] == 1;
}

}  // namespace colmap
