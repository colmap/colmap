// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/util/file.h"

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
  THROW_CHECK(!ext.empty());
  THROW_CHECK_EQ(ext.at(0), '.');
  std::string ext_lower = ext;
  StringToLower(&ext_lower);
  if (file_name.size() >= ext_lower.size() &&
      file_name.substr(file_name.size() - ext_lower.size(), ext_lower.size()) ==
          ext_lower) {
    return true;
  }
  return false;
}

void SplitFileExtension(const std::string& path,
                        std::string* root,
                        std::string* ext) {
  const auto parts = StringSplit(path, ".");
  THROW_CHECK_GT(parts.size(), 0);
  if (parts.size() == 1) {
    *root = parts[0];
    *ext = "";
  } else {
    *root = "";
    for (size_t i = 0; i < parts.size() - 1; ++i) {
      *root += parts[i] + ".";
    }
    *root = root->substr(0, root->length() - 1);
    if (parts.back() == "") {
      *ext = "";
    } else {
      *ext = "." + parts.back();
    }
  }
}

void FileCopy(const std::string& src_path,
              const std::string& dst_path,
              CopyType type) {
  switch (type) {
    case CopyType::COPY:
      std::filesystem::copy_file(src_path, dst_path);
      break;
    case CopyType::HARD_LINK:
      std::filesystem::create_hard_link(src_path, dst_path);
      break;
    case CopyType::SOFT_LINK:
      std::filesystem::create_symlink(src_path, dst_path);
      break;
  }
}

bool ExistsFile(const std::string& path) {
  return std::filesystem::is_regular_file(path);
}

bool ExistsDir(const std::string& path) {
  return std::filesystem::is_directory(path);
}

bool ExistsPath(const std::string& path) {
  return std::filesystem::exists(path);
}

void CreateDirIfNotExists(const std::string& path, bool recursive) {
  if (ExistsDir(path)) {
    return;
  }
  if (recursive) {
    THROW_CHECK(std::filesystem::create_directories(path));
  } else {
    THROW_CHECK(std::filesystem::create_directory(path));
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

std::string GetParentDir(const std::string& path) {
  return std::filesystem::path(path).parent_path().string();
}

std::vector<std::string> GetFileList(const std::string& path) {
  std::vector<std::string> file_list;
  for (auto it = std::filesystem::directory_iterator(path);
       it != std::filesystem::directory_iterator();
       ++it) {
    if (std::filesystem::is_regular_file(*it)) {
      const std::filesystem::path file_path = *it;
      file_list.push_back(file_path.string());
    }
  }
  return file_list;
}

std::vector<std::string> GetRecursiveFileList(const std::string& path) {
  std::vector<std::string> file_list;
  for (auto it = std::filesystem::recursive_directory_iterator(path);
       it != std::filesystem::recursive_directory_iterator();
       ++it) {
    if (std::filesystem::is_regular_file(*it)) {
      const std::filesystem::path file_path = *it;
      file_list.push_back(file_path.string());
    }
  }
  return file_list;
}

std::vector<std::string> GetDirList(const std::string& path) {
  std::vector<std::string> dir_list;
  for (auto it = std::filesystem::directory_iterator(path);
       it != std::filesystem::directory_iterator();
       ++it) {
    if (std::filesystem::is_directory(*it)) {
      const std::filesystem::path dir_path = *it;
      dir_list.push_back(dir_path.string());
    }
  }
  return dir_list;
}

std::vector<std::string> GetRecursiveDirList(const std::string& path) {
  std::vector<std::string> dir_list;
  for (auto it = std::filesystem::recursive_directory_iterator(path);
       it != std::filesystem::recursive_directory_iterator();
       ++it) {
    if (std::filesystem::is_directory(*it)) {
      const std::filesystem::path dir_path = *it;
      dir_list.push_back(dir_path.string());
    }
  }
  return dir_list;
}

size_t GetFileSize(const std::string& path) {
  std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  return file.tellg();
}

std::vector<std::string> ReadTextFileLines(const std::string& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    lines.push_back(line);
  }

  return lines;
}

}  // namespace colmap
