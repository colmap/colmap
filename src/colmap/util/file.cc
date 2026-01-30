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

#include "colmap/util/file.h"

#include "colmap/util/logging.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

#ifdef COLMAP_DOWNLOAD_ENABLED
#include <curl/curl.h>
#if defined(COLMAP_USE_CRYPTOPP)
#include <cryptopp/sha.h>
#elif defined(COLMAP_USE_OPENSSL)
#include <openssl/sha.h>
#else
#error "No crypto library defined"
#endif
#endif

#ifndef _MSC_VER
extern "C" {
extern char** environ;
}
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

bool HasFileExtension(const std::filesystem::path& file_name,
                      const std::string& ext) {
  THROW_CHECK(!ext.empty());
  THROW_CHECK_EQ(ext.at(0), '.');
  std::string ext_lower = ext;
  StringToLower(&ext_lower);
  return file_name.extension() == ext_lower;
}

std::filesystem::path AddFileExtension(std::filesystem::path path,
                                       const std::string& ext) {
  path += ext;
  return path;
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

void FileCopy(const std::filesystem::path& src_path,
              const std::filesystem::path& dst_path,
              FileCopyType type) {
  switch (type) {
    case FileCopyType::COPY:
      std::filesystem::copy_file(src_path, dst_path);
      break;
    case FileCopyType::HARD_LINK:
      std::filesystem::create_hard_link(src_path, dst_path);
      break;
    case FileCopyType::SOFT_LINK:
      std::filesystem::create_symlink(src_path, dst_path);
      break;
  }
}

bool ExistsFile(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(path);
}

bool ExistsDir(const std::filesystem::path& path) {
  return std::filesystem::is_directory(path);
}

bool ExistsPath(const std::filesystem::path& path) {
  return std::filesystem::exists(path);
}

void CreateDirIfNotExists(const std::filesystem::path& path, bool recursive) {
  if (ExistsDir(path)) {
    return;
  }
  if (recursive) {
    THROW_CHECK(std::filesystem::create_directories(path));
  } else {
    THROW_CHECK(std::filesystem::create_directory(path));
  }
}

std::filesystem::path GetPathBaseName(const std::filesystem::path& path) {
  const std::filesystem::path fs_path(NormalizePath(path));
  if (fs_path.has_filename()) {
    return fs_path.filename();
  } else {  // It is a directory.
    return fs_path.parent_path().filename();
  }
}

std::filesystem::path GetParentDir(const std::filesystem::path& path) {
  return path.parent_path();
}

std::string NormalizePath(const std::filesystem::path& path) {
  std::string normalized_path = path.lexically_normal().string();
  if constexpr (std::filesystem::path::preferred_separator == '\\') {
    normalized_path = StringReplace(normalized_path, "\\", "/");
  }
  return normalized_path;
}

std::string GetNormalizedRelativePath(const std::filesystem::path& full_path,
                                      const std::filesystem::path& base_path) {
  return NormalizePath(std::filesystem::relative(full_path, base_path));
}

std::vector<std::filesystem::path> GetRecursiveFileList(
    const std::filesystem::path& path) {
  std::vector<std::filesystem::path> file_list;
  for (auto it = std::filesystem::recursive_directory_iterator(path);
       it != std::filesystem::recursive_directory_iterator();
       ++it) {
    if (std::filesystem::is_regular_file(*it)) {
      const std::filesystem::path file_path = *it;
      file_list.push_back(file_path);
    }
  }
  return file_list;
}

std::vector<std::filesystem::path> GetDirList(
    const std::filesystem::path& path) {
  std::vector<std::filesystem::path> dir_list;
  for (auto it = std::filesystem::directory_iterator(path);
       it != std::filesystem::directory_iterator();
       ++it) {
    if (std::filesystem::is_directory(*it)) {
      const std::filesystem::path dir_path = *it;
      dir_list.push_back(dir_path);
    }
  }
  return dir_list;
}

namespace {

std::optional<std::string> GetEnvSafe(const char* key) {
#ifdef _MSC_VER
  size_t size = 0;
  getenv_s(&size, nullptr, 0, key);
  if (size == 0) {
    return std::nullopt;
  }
  std::string value(size, ' ');
  getenv_s(&size, value.data(), size, key);
  // getenv_s returns a null-terminated string, so we need to remove the
  // trailing null character in our std::string.
  THROW_CHECK_EQ(value.back(), '\0');
  return value.substr(0, size - 1);
#else
  // Non-MSVC replacement for std::getenv_s. The safe variant
  // std::getenv_s is not available on all platforms, unfortunately.
  // Stores environment variables as: "key1=value1", "key2=value2", ..., null
  char** env = environ;
  const std::string_view key_sv(key);
  for (; *env; ++env) {
    const std::string_view key_value(*env);
    if (key_sv.size() < key_value.size() &&
        key_value.substr(0, key_sv.size()) == key_sv &&
        key_value[key_sv.size()] == '=') {
      return std::string(key_value.substr(
          key_sv.size() + 1, key_value.size() - key_sv.size() - 1));
    }
  }
  return std::nullopt;
#endif
}

}  // namespace

std::optional<std::filesystem::path> HomeDir() {
#ifdef _MSC_VER
  std::optional<std::string> userprofile = GetEnvSafe("USERPROFILE");
  if (userprofile.has_value()) {
    return *userprofile;
  }
  const std::optional<std::string> homedrive = GetEnvSafe("HOMEDRIVE");
  const std::optional<std::string> homepath = GetEnvSafe("HOMEPATH");
  if (!homedrive.has_value() || !homepath.has_value()) {
    return std::nullopt;
  }
  return std::filesystem::path(*homedrive) / std::filesystem::path(*homepath);
#else
  std::optional<std::string> home = GetEnvSafe("HOME");
  if (!home.has_value()) {
    return std::nullopt;
  }
  return *home;
#endif
}

void ReadBinaryBlob(const std::filesystem::path& path,
                    std::vector<char>* data) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  THROW_CHECK_FILE_OPEN(file, path);
  file.seekg(0, std::ios::end);
  const size_t num_bytes = file.tellg();
  data->resize(num_bytes);
  file.seekg(0, std::ios::beg);
  file.read(data->data(), num_bytes);
}

void WriteBinaryBlob(const std::filesystem::path& path,
                     const span<const char>& data) {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  file.write(data.begin(), data.size());
}

std::vector<std::string> ReadTextFileLines(const std::filesystem::path& path) {
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

bool IsURI(const std::string& uri) {
  return StringStartsWith(uri, "http://") ||
         StringStartsWith(uri, "https://") || StringStartsWith(uri, "file://");
}

#ifdef COLMAP_DOWNLOAD_ENABLED

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
  VLOG(2) << "Downloading file from: " << url;

  CurlHandle handle;
  THROW_CHECK_NOTNULL(handle.ptr);

  curl_easy_setopt(handle.ptr, CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle.ptr, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEFUNCTION, &WriteCurlData);
  std::ostringstream data_stream;
  curl_easy_setopt(handle.ptr, CURLOPT_WRITEDATA, &data_stream);

  // Respect SSL_CERT_FILE and SSL_CERT_DIR environment variables for
  // cross-distribution compatibility (e.g., Ubuntu vs RHEL-based systems).
  // This can be an issue in pycolmap, where we build cross-platform for Linux
  // using RHEL while users may run pycolmap on Ubuntu, etc.
  const std::optional<std::string> ssl_cert_file = GetEnvSafe("SSL_CERT_FILE");
  if (ssl_cert_file.has_value() && !ssl_cert_file->empty()) {
    VLOG(2) << "Using SSL_CERT_FILE: " << *ssl_cert_file;
    curl_easy_setopt(handle.ptr, CURLOPT_CAINFO, ssl_cert_file->c_str());
  }
  const std::optional<std::string> ssl_cert_dir = GetEnvSafe("SSL_CERT_DIR");
  if (ssl_cert_dir.has_value() && !ssl_cert_dir->empty()) {
    VLOG(2) << "Using SSL_CERT_DIR: " << *ssl_cert_dir;
    curl_easy_setopt(handle.ptr, CURLOPT_CAPATH, ssl_cert_dir->c_str());
  }

  const CURLcode code = curl_easy_perform(handle.ptr);
  if (code != CURLE_OK) {
    if (code == CURLE_SSL_CACERT_BADFILE || code == CURLE_SSL_CACERT) {
      LOG(ERROR) << "Curl SSL certificate error (code " << code
                 << "). Try setting SSL_CERT_FILE to point to your system's "
                    "CA certificate bundle (e.g., "
                    "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt on "
                    "Ubuntu/Debian).";
    } else {
      VLOG(2) << "Curl failed to perform request with code: " << code;
    }
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
}

namespace {

std::string SHA256DigestToHex(span<unsigned char> digest) {
  std::ostringstream hex;
  for (const auto c : digest) {
    hex << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned int>(c);
  }
  return hex.str();
}

}  // namespace

#if defined(COLMAP_USE_CRYPTOPP)

std::string ComputeSHA256(const std::string_view& str) {
  CryptoPP::byte digest[CryptoPP::SHA256::DIGESTSIZE];
  CryptoPP::SHA256().CalculateDigest(
      digest, reinterpret_cast<const CryptoPP::byte*>(str.data()), str.size());
  return SHA256DigestToHex({digest, CryptoPP::SHA256::DIGESTSIZE});
}

#elif defined(COLMAP_USE_OPENSSL)

std::string ComputeSHA256(const std::string_view& str) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(
      reinterpret_cast<const unsigned char*>(str.data()), str.size(), digest);
  return SHA256DigestToHex({digest, SHA256_DIGEST_LENGTH});
}

#else
#error "No crypto library defined"
#endif

namespace {

std::optional<std::filesystem::path> download_cache_dir_overwrite;

}

std::string DownloadAndCacheFile(const std::string& uri) {
  const std::vector<std::string> parts = StringSplit(uri, ";");
  THROW_CHECK_EQ(parts.size(), 3)
      << "Invalid URI format. Expected: <url>;<name>;<sha256>";

  const std::string& url = parts[0];
  THROW_CHECK(!url.empty());
  const std::string& name = parts[1];
  THROW_CHECK(!name.empty());
  const std::string& sha256 = parts[2];
  THROW_CHECK_EQ(sha256.size(), 64);

  std::filesystem::path download_cache_dir;
  if (download_cache_dir_overwrite.has_value()) {
    download_cache_dir = *download_cache_dir_overwrite;
  } else {
    const std::optional<std::filesystem::path> home_dir = HomeDir();
    THROW_CHECK(home_dir.has_value());
    download_cache_dir = *home_dir / ".cache" / "colmap";
  }

  if (!std::filesystem::exists(download_cache_dir)) {
    VLOG(2) << "Creating download cache directory: " << download_cache_dir;
    THROW_CHECK(std::filesystem::create_directories(download_cache_dir));
  }

  const auto path = download_cache_dir / (sha256 + "-" + name);

  if (std::filesystem::exists(path)) {
    VLOG(2) << "File already downloaded. Skipping download.";
    std::vector<char> blob;
    ReadBinaryBlob(path.string(), &blob);
    THROW_CHECK_EQ(ComputeSHA256({blob.data(), blob.size()}), sha256)
        << "The cached file does not match the expected SHA256";
  } else {
    LOG(INFO) << "Downloading file from: " << url;
    const std::optional<std::string> blob = DownloadFile(url);
    THROW_CHECK(blob.has_value()) << "Failed to download file";
    THROW_CHECK_EQ(ComputeSHA256({blob->data(), blob->size()}), sha256)
        << "The downloaded file does not match the expected SHA256";
    LOG(INFO) << "Caching file at: " << path;
    WriteBinaryBlob(path.string(), {blob->data(), blob->size()});
  }

  return path.string();
}

void OverwriteDownloadCacheDir(std::filesystem::path path) {
  download_cache_dir_overwrite = std::move(path);
}

#endif  // COLMAP_DOWNLOAD_ENABLED

std::filesystem::path MaybeDownloadAndCacheFile(const std::string& uri) {
  if (!IsURI(uri)) {
    return uri;
  }
#ifdef COLMAP_DOWNLOAD_ENABLED
  return DownloadAndCacheFile(uri);
#else
  throw std::runtime_error("COLMAP was compiled without download support");
#endif
}

}  // namespace colmap
