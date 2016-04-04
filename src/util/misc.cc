// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include <fstream>
#include <sstream>

#include <boost/algorithm/string.hpp>

#include "util/logging.h"

namespace colmap {

std::string EnsureTrailingSlash(const std::string& str) {
  if (str.length() > 0) {
    if (str.at(str.length() - 1) != '/') {
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

void PrintHeading1(const std::string& heading) {
  std::cout << std::endl << std::string(78, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(78, '=') << std::endl << std::endl;
}

void PrintHeading2(const std::string& heading) {
  std::cout << std::endl << heading << std::endl;
  std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
}

std::string StringReplace(const std::string& str, const std::string& old_str,
                          const std::string& new_str) {
  if (old_str.empty()) {
    return str;
  }
  size_t position = 0;
  std::string mod_str = str;
  while ((position = mod_str.find(old_str, position)) != std::string::npos) {
    mod_str.replace(position, old_str.size(), new_str);
    position += new_str.size();
  }
  return mod_str;
}

std::vector<std::string> StringSplit(const std::string& str,
                                     const std::string& delim) {
  std::vector<std::string> elems;
  boost::split(elems, str, boost::is_any_of(delim), boost::token_compress_on);
  return elems;
}

}  // namespace colmap
