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

#include "colmap/util/logging.h"

#include <iomanip>
#include <mutex>
#include <sstream>

#ifdef COLMAP_AUTO_DOWNLOAD_ENABLED
#include <curl/curl.h>
#include <openssl/evp.h>
#endif

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

void ReadBinaryBlob(const std::string& path, std::vector<char>* data) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  THROW_CHECK_FILE_OPEN(file, path);
  file.seekg(0, std::ios::end);
  const size_t num_bytes = file.tellg();
  data->resize(num_bytes);
  file.seekg(0, std::ios::beg);
  file.read(data->data(), num_bytes);
}

void WriteBinaryBlob(const std::string& path, const span<const char>& data) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  file.write(data.begin(), data.size());
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

namespace {

size_t WriteCurlData(char* buf,
                     size_t size,
                     size_t nmemb,
                     std::ostringstream* data_stream) {
  *data_stream << std::string_view(buf, size * nmemb);
  return size * nmemb;
}

struct CurlHandle {
  CurlHandle() {
    static std::once_flag global_curl_init;
    std::call_once(global_curl_init,
                   []() { curl_global_init(CURL_GLOBAL_ALL); });

    ptr = curl_easy_init();
  }

  ~CurlHandle() { curl_easy_cleanup(ptr); }

  CURL* ptr;
};

}  // namespace

std::optional<std::string> DownloadFile(const std::string& url) {
#ifdef COLMAP_AUTO_DOWNLOAD_ENABLED
  VLOG(2) << "Downloading file from: " << url;

  CurlHandle handle;
  THROW_CHECK_NOTNULL(handle.ptr);

  curl_easy_setopt(handle.ptr, CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle.ptr, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEFUNCTION, &WriteCurlData);
  std::ostringstream data_stream;
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEDATA, &data_stream);

  const CURLcode code = curl_easy_perform(handle.ptr);
  if (code != CURLE_OK) {
    VLOG(2) << "Curl failed to perform request with code: " << code;
    return std::nullopt;
  }

  long response_code = 0;
  curl_easy_getinfo(handle.ptr, CURLINFO_RESPONSE_CODE, &response_code);
  if (response_code != 0 && (response_code < 200 || response_code >= 300)) {
    VLOG(2) << "Request failed with status: " << response_code;
    return std::nullopt;
  }

  std::string data_str = data_stream.str();
  VLOG(2) << "Downloaded " << data_str.size() << " bytes";

  return data_str;
#else
  LOG(ERROR) << "COLMAP was compiled without Curl support.";
  return std::nullopt;
#endif
}

std::string ComputeSHA256(const std::string_view& str) {
  auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);

  unsigned int hash_length = 0;
  unsigned char hash[EVP_MAX_MD_SIZE];

  EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr);
  EVP_DigestUpdate(context.get(), str.data(), str.size());
  EVP_DigestFinal_ex(context.get(), hash, &hash_length);

  std::ostringstream digest;
  for (unsigned int i = 0; i < hash_length; ++i) {
    digest << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned int>(hash[i]);
  }
  return digest.str();
}

}  // namespace colmap
