//
//  httplib.h
//
//  Copyright (c) 2024 Yuji Hirose. All rights reserved.
//  MIT License
//

#ifndef CPPHTTPLIB_HTTPLIB_H
#define CPPHTTPLIB_HTTPLIB_H

#define CPPHTTPLIB_VERSION "0.18.3"

/*
 * Configuration
 */

#ifndef CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND
#define CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND 5
#endif

#ifndef CPPHTTPLIB_KEEPALIVE_TIMEOUT_CHECK_INTERVAL_USECOND
#define CPPHTTPLIB_KEEPALIVE_TIMEOUT_CHECK_INTERVAL_USECOND 10000
#endif

#ifndef CPPHTTPLIB_KEEPALIVE_MAX_COUNT
#define CPPHTTPLIB_KEEPALIVE_MAX_COUNT 100
#endif

#ifndef CPPHTTPLIB_CONNECTION_TIMEOUT_SECOND
#define CPPHTTPLIB_CONNECTION_TIMEOUT_SECOND 300
#endif

#ifndef CPPHTTPLIB_CONNECTION_TIMEOUT_USECOND
#define CPPHTTPLIB_CONNECTION_TIMEOUT_USECOND 0
#endif

#ifndef CPPHTTPLIB_SERVER_READ_TIMEOUT_SECOND
#define CPPHTTPLIB_SERVER_READ_TIMEOUT_SECOND 5
#endif

#ifndef CPPHTTPLIB_SERVER_READ_TIMEOUT_USECOND
#define CPPHTTPLIB_SERVER_READ_TIMEOUT_USECOND 0
#endif

#ifndef CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND
#define CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND 5
#endif

#ifndef CPPHTTPLIB_SERVER_WRITE_TIMEOUT_USECOND
#define CPPHTTPLIB_SERVER_WRITE_TIMEOUT_USECOND 0
#endif

#ifndef CPPHTTPLIB_CLIENT_READ_TIMEOUT_SECOND
#define CPPHTTPLIB_CLIENT_READ_TIMEOUT_SECOND 300
#endif

#ifndef CPPHTTPLIB_CLIENT_READ_TIMEOUT_USECOND
#define CPPHTTPLIB_CLIENT_READ_TIMEOUT_USECOND 0
#endif

#ifndef CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_SECOND
#define CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_SECOND 5
#endif

#ifndef CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_USECOND
#define CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_USECOND 0
#endif

#ifndef CPPHTTPLIB_IDLE_INTERVAL_SECOND
#define CPPHTTPLIB_IDLE_INTERVAL_SECOND 0
#endif

#ifndef CPPHTTPLIB_IDLE_INTERVAL_USECOND
#ifdef _WIN32
#define CPPHTTPLIB_IDLE_INTERVAL_USECOND 10000
#else
#define CPPHTTPLIB_IDLE_INTERVAL_USECOND 0
#endif
#endif

#ifndef CPPHTTPLIB_REQUEST_URI_MAX_LENGTH
#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH 8192
#endif

#ifndef CPPHTTPLIB_HEADER_MAX_LENGTH
#define CPPHTTPLIB_HEADER_MAX_LENGTH 8192
#endif

#ifndef CPPHTTPLIB_REDIRECT_MAX_COUNT
#define CPPHTTPLIB_REDIRECT_MAX_COUNT 20
#endif

#ifndef CPPHTTPLIB_MULTIPART_FORM_DATA_FILE_MAX_COUNT
#define CPPHTTPLIB_MULTIPART_FORM_DATA_FILE_MAX_COUNT 1024
#endif

#ifndef CPPHTTPLIB_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_PAYLOAD_MAX_LENGTH ((std::numeric_limits<size_t>::max)())
#endif

#ifndef CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 8192
#endif

#ifndef CPPHTTPLIB_RANGE_MAX_COUNT
#define CPPHTTPLIB_RANGE_MAX_COUNT 1024
#endif

#ifndef CPPHTTPLIB_TCP_NODELAY
#define CPPHTTPLIB_TCP_NODELAY false
#endif

#ifndef CPPHTTPLIB_IPV6_V6ONLY
#define CPPHTTPLIB_IPV6_V6ONLY false
#endif

#ifndef CPPHTTPLIB_RECV_BUFSIZ
#define CPPHTTPLIB_RECV_BUFSIZ size_t(16384u)
#endif

#ifndef CPPHTTPLIB_COMPRESSION_BUFSIZ
#define CPPHTTPLIB_COMPRESSION_BUFSIZ size_t(16384u)
#endif

#ifndef CPPHTTPLIB_THREAD_POOL_COUNT
#define CPPHTTPLIB_THREAD_POOL_COUNT                                           \
  ((std::max)(8u, std::thread::hardware_concurrency() > 0                      \
                      ? std::thread::hardware_concurrency() - 1                \
                      : 0))
#endif

#ifndef CPPHTTPLIB_RECV_FLAGS
#define CPPHTTPLIB_RECV_FLAGS 0
#endif

#ifndef CPPHTTPLIB_SEND_FLAGS
#define CPPHTTPLIB_SEND_FLAGS 0
#endif

#ifndef CPPHTTPLIB_LISTEN_BACKLOG
#define CPPHTTPLIB_LISTEN_BACKLOG 5
#endif

/*
 * Headers
 */

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif //_CRT_NONSTDC_NO_DEPRECATE

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#error Sorry, Visual Studio versions prior to 2015 are not supported
#endif

#pragma comment(lib, "ws2_32.lib")

#ifdef _WIN64
using ssize_t = __int64;
#else
using ssize_t = long;
#endif
#endif // _MSC_VER

#ifndef S_ISREG
#define S_ISREG(m) (((m) & S_IFREG) == S_IFREG)
#endif // S_ISREG

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFDIR) == S_IFDIR)
#endif // S_ISDIR

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef WSA_FLAG_NO_HANDLE_INHERIT
#define WSA_FLAG_NO_HANDLE_INHERIT 0x80
#endif

using socket_t = SOCKET;
#ifdef CPPHTTPLIB_USE_POLL
#define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
#endif

#else // not _WIN32

#include <arpa/inet.h>
#if !defined(_AIX) && !defined(__MVS__)
#include <ifaddrs.h>
#endif
#ifdef __MVS__
#include <strings.h>
#ifndef NI_MAXHOST
#define NI_MAXHOST 1025
#endif
#endif
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#ifdef __linux__
#include <resolv.h>
#endif
#include <netinet/tcp.h>
#ifdef CPPHTTPLIB_USE_POLL
#include <poll.h>
#endif
#include <csignal>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif //_WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#ifdef _WIN32
#include <wincrypt.h>

// these are defined in wincrypt.h and it breaks compilation if BoringSSL is
// used
#undef X509_NAME
#undef X509_CERT_PAIR
#undef X509_EXTENSIONS
#undef PKCS7_SIGNER_INFO

#ifdef _MSC_VER
#pragma comment(lib, "crypt32.lib")
#endif
#elif defined(CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN) && defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#endif // TARGET_OS_OSX
#endif // _WIN32

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#if defined(_WIN32) && defined(OPENSSL_USE_APPLINK)
#include <openssl/applink.c>
#endif

#include <iostream>
#include <sstream>

#if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
#if OPENSSL_VERSION_NUMBER < 0x1010107f
#error Please use OpenSSL or a current version of BoringSSL
#endif
#define SSL_get1_peer_certificate SSL_get_peer_certificate
#elif OPENSSL_VERSION_NUMBER < 0x30000000L
#error Sorry, OpenSSL versions prior to 3.0.0 are not supported
#endif

#endif

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
#include <zlib.h>
#endif

#ifdef CPPHTTPLIB_BROTLI_SUPPORT
#include <brotli/decode.h>
#include <brotli/encode.h>
#endif

/*
 * Declaration
 */
namespace httplib {

namespace detail {

/*
 * Backport std::make_unique from C++14.
 *
 * NOTE: This code came up with the following stackoverflow post:
 * https://stackoverflow.com/questions/10149840/c-arrays-and-make-unique
 *
 */

template <class T, class... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(std::size_t n) {
  typedef typename std::remove_extent<T>::type RT;
  return std::unique_ptr<T>(new RT[n]);
}

namespace case_ignore {

inline unsigned char to_lower(int c) {
  const static unsigned char table[256] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
      107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
      122, 91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
      165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 224, 225, 226,
      227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
      242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, 224,
      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
      240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
      255,
  };
  return table[(unsigned char)(char)c];
}

inline bool equal(const std::string &a, const std::string &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char ca, char cb) {
           return to_lower(ca) == to_lower(cb);
         });
}

struct equal_to {
  bool operator()(const std::string &a, const std::string &b) const {
    return equal(a, b);
  }
};

struct hash {
  size_t operator()(const std::string &key) const {
    return hash_core(key.data(), key.size(), 0);
  }

  size_t hash_core(const char *s, size_t l, size_t h) const {
    return (l == 0) ? h
                    : hash_core(s + 1, l - 1,
                                // Unsets the 6 high bits of h, therefore no
                                // overflow happens
                                (((std::numeric_limits<size_t>::max)() >> 6) &
                                 h * 33) ^
                                    static_cast<unsigned char>(to_lower(*s)));
  }
};

} // namespace case_ignore

// This is based on
// "http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4189".

struct scope_exit {
  explicit scope_exit(std::function<void(void)> &&f)
      : exit_function(std::move(f)), execute_on_destruction{true} {}

  scope_exit(scope_exit &&rhs) noexcept
      : exit_function(std::move(rhs.exit_function)),
        execute_on_destruction{rhs.execute_on_destruction} {
    rhs.release();
  }

  ~scope_exit() {
    if (execute_on_destruction) { this->exit_function(); }
  }

  void release() { this->execute_on_destruction = false; }

private:
  scope_exit(const scope_exit &) = delete;
  void operator=(const scope_exit &) = delete;
  scope_exit &operator=(scope_exit &&) = delete;

  std::function<void(void)> exit_function;
  bool execute_on_destruction;
};

} // namespace detail

enum StatusCode {
  // Information responses
  Continue_100 = 100,
  SwitchingProtocol_101 = 101,
  Processing_102 = 102,
  EarlyHints_103 = 103,

  // Successful responses
  OK_200 = 200,
  Created_201 = 201,
  Accepted_202 = 202,
  NonAuthoritativeInformation_203 = 203,
  NoContent_204 = 204,
  ResetContent_205 = 205,
  PartialContent_206 = 206,
  MultiStatus_207 = 207,
  AlreadyReported_208 = 208,
  IMUsed_226 = 226,

  // Redirection messages
  MultipleChoices_300 = 300,
  MovedPermanently_301 = 301,
  Found_302 = 302,
  SeeOther_303 = 303,
  NotModified_304 = 304,
  UseProxy_305 = 305,
  unused_306 = 306,
  TemporaryRedirect_307 = 307,
  PermanentRedirect_308 = 308,

  // Client error responses
  BadRequest_400 = 400,
  Unauthorized_401 = 401,
  PaymentRequired_402 = 402,
  Forbidden_403 = 403,
  NotFound_404 = 404,
  MethodNotAllowed_405 = 405,
  NotAcceptable_406 = 406,
  ProxyAuthenticationRequired_407 = 407,
  RequestTimeout_408 = 408,
  Conflict_409 = 409,
  Gone_410 = 410,
  LengthRequired_411 = 411,
  PreconditionFailed_412 = 412,
  PayloadTooLarge_413 = 413,
  UriTooLong_414 = 414,
  UnsupportedMediaType_415 = 415,
  RangeNotSatisfiable_416 = 416,
  ExpectationFailed_417 = 417,
  ImATeapot_418 = 418,
  MisdirectedRequest_421 = 421,
  UnprocessableContent_422 = 422,
  Locked_423 = 423,
  FailedDependency_424 = 424,
  TooEarly_425 = 425,
  UpgradeRequired_426 = 426,
  PreconditionRequired_428 = 428,
  TooManyRequests_429 = 429,
  RequestHeaderFieldsTooLarge_431 = 431,
  UnavailableForLegalReasons_451 = 451,

  // Server error responses
  InternalServerError_500 = 500,
  NotImplemented_501 = 501,
  BadGateway_502 = 502,
  ServiceUnavailable_503 = 503,
  GatewayTimeout_504 = 504,
  HttpVersionNotSupported_505 = 505,
  VariantAlsoNegotiates_506 = 506,
  InsufficientStorage_507 = 507,
  LoopDetected_508 = 508,
  NotExtended_510 = 510,
  NetworkAuthenticationRequired_511 = 511,
};

using Headers =
    std::unordered_multimap<std::string, std::string, detail::case_ignore::hash,
                            detail::case_ignore::equal_to>;

using Params = std::multimap<std::string, std::string>;
using Match = std::smatch;

using Progress = std::function<bool(uint64_t current, uint64_t total)>;

struct Response;
using ResponseHandler = std::function<bool(const Response &response)>;

struct MultipartFormData {
  std::string name;
  std::string content;
  std::string filename;
  std::string content_type;
};
using MultipartFormDataItems = std::vector<MultipartFormData>;
using MultipartFormDataMap = std::multimap<std::string, MultipartFormData>;

class DataSink {
public:
  DataSink() : os(&sb_), sb_(*this) {}

  DataSink(const DataSink &) = delete;
  DataSink &operator=(const DataSink &) = delete;
  DataSink(DataSink &&) = delete;
  DataSink &operator=(DataSink &&) = delete;

  std::function<bool(const char *data, size_t data_len)> write;
  std::function<bool()> is_writable;
  std::function<void()> done;
  std::function<void(const Headers &trailer)> done_with_trailer;
  std::ostream os;

private:
  class data_sink_streambuf final : public std::streambuf {
  public:
    explicit data_sink_streambuf(DataSink &sink) : sink_(sink) {}

  protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override {
      sink_.write(s, static_cast<size_t>(n));
      return n;
    }

  private:
    DataSink &sink_;
  };

  data_sink_streambuf sb_;
};

using ContentProvider =
    std::function<bool(size_t offset, size_t length, DataSink &sink)>;

using ContentProviderWithoutLength =
    std::function<bool(size_t offset, DataSink &sink)>;

using ContentProviderResourceReleaser = std::function<void(bool success)>;

struct MultipartFormDataProvider {
  std::string name;
  ContentProviderWithoutLength provider;
  std::string filename;
  std::string content_type;
};
using MultipartFormDataProviderItems = std::vector<MultipartFormDataProvider>;

using ContentReceiverWithProgress =
    std::function<bool(const char *data, size_t data_length, uint64_t offset,
                       uint64_t total_length)>;

using ContentReceiver =
    std::function<bool(const char *data, size_t data_length)>;

using MultipartContentHeader =
    std::function<bool(const MultipartFormData &file)>;

class ContentReader {
public:
  using Reader = std::function<bool(ContentReceiver receiver)>;
  using MultipartReader = std::function<bool(MultipartContentHeader header,
                                             ContentReceiver receiver)>;

  ContentReader(Reader reader, MultipartReader multipart_reader)
      : reader_(std::move(reader)),
        multipart_reader_(std::move(multipart_reader)) {}

  bool operator()(MultipartContentHeader header,
                  ContentReceiver receiver) const {
    return multipart_reader_(std::move(header), std::move(receiver));
  }

  bool operator()(ContentReceiver receiver) const {
    return reader_(std::move(receiver));
  }

  Reader reader_;
  MultipartReader multipart_reader_;
};

using Range = std::pair<ssize_t, ssize_t>;
using Ranges = std::vector<Range>;

struct Request {
  std::string method;
  std::string path;
  Params params;
  Headers headers;
  std::string body;

  std::string remote_addr;
  int remote_port = -1;
  std::string local_addr;
  int local_port = -1;

  // for server
  std::string version;
  std::string target;
  MultipartFormDataMap files;
  Ranges ranges;
  Match matches;
  std::unordered_map<std::string, std::string> path_params;

  // for client
  ResponseHandler response_handler;
  ContentReceiverWithProgress content_receiver;
  Progress progress;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  const SSL *ssl = nullptr;
#endif

  bool has_header(const std::string &key) const;
  std::string get_header_value(const std::string &key, const char *def = "",
                               size_t id = 0) const;
  uint64_t get_header_value_u64(const std::string &key, uint64_t def = 0,
                                size_t id = 0) const;
  size_t get_header_value_count(const std::string &key) const;
  void set_header(const std::string &key, const std::string &val);

  bool has_param(const std::string &key) const;
  std::string get_param_value(const std::string &key, size_t id = 0) const;
  size_t get_param_value_count(const std::string &key) const;

  bool is_multipart_form_data() const;

  bool has_file(const std::string &key) const;
  MultipartFormData get_file_value(const std::string &key) const;
  std::vector<MultipartFormData> get_file_values(const std::string &key) const;

  // private members...
  size_t redirect_count_ = CPPHTTPLIB_REDIRECT_MAX_COUNT;
  size_t content_length_ = 0;
  ContentProvider content_provider_;
  bool is_chunked_content_provider_ = false;
  size_t authorization_count_ = 0;
};

struct Response {
  std::string version;
  int status = -1;
  std::string reason;
  Headers headers;
  std::string body;
  std::string location; // Redirect location

  bool has_header(const std::string &key) const;
  std::string get_header_value(const std::string &key, const char *def = "",
                               size_t id = 0) const;
  uint64_t get_header_value_u64(const std::string &key, uint64_t def = 0,
                                size_t id = 0) const;
  size_t get_header_value_count(const std::string &key) const;
  void set_header(const std::string &key, const std::string &val);

  void set_redirect(const std::string &url, int status = StatusCode::Found_302);
  void set_content(const char *s, size_t n, const std::string &content_type);
  void set_content(const std::string &s, const std::string &content_type);
  void set_content(std::string &&s, const std::string &content_type);

  void set_content_provider(
      size_t length, const std::string &content_type, ContentProvider provider,
      ContentProviderResourceReleaser resource_releaser = nullptr);

  void set_content_provider(
      const std::string &content_type, ContentProviderWithoutLength provider,
      ContentProviderResourceReleaser resource_releaser = nullptr);

  void set_chunked_content_provider(
      const std::string &content_type, ContentProviderWithoutLength provider,
      ContentProviderResourceReleaser resource_releaser = nullptr);

  void set_file_content(const std::string &path,
                        const std::string &content_type);
  void set_file_content(const std::string &path);

  Response() = default;
  Response(const Response &) = default;
  Response &operator=(const Response &) = default;
  Response(Response &&) = default;
  Response &operator=(Response &&) = default;
  ~Response() {
    if (content_provider_resource_releaser_) {
      content_provider_resource_releaser_(content_provider_success_);
    }
  }

  // private members...
  size_t content_length_ = 0;
  ContentProvider content_provider_;
  ContentProviderResourceReleaser content_provider_resource_releaser_;
  bool is_chunked_content_provider_ = false;
  bool content_provider_success_ = false;
  std::string file_content_path_;
  std::string file_content_content_type_;
};

class Stream {
public:
  virtual ~Stream() = default;

  virtual bool is_readable() const = 0;
  virtual bool is_writable() const = 0;

  virtual ssize_t read(char *ptr, size_t size) = 0;
  virtual ssize_t write(const char *ptr, size_t size) = 0;
  virtual void get_remote_ip_and_port(std::string &ip, int &port) const = 0;
  virtual void get_local_ip_and_port(std::string &ip, int &port) const = 0;
  virtual socket_t socket() const = 0;

  ssize_t write(const char *ptr);
  ssize_t write(const std::string &s);
};

class TaskQueue {
public:
  TaskQueue() = default;
  virtual ~TaskQueue() = default;

  virtual bool enqueue(std::function<void()> fn) = 0;
  virtual void shutdown() = 0;

  virtual void on_idle() {}
};

class ThreadPool final : public TaskQueue {
public:
  explicit ThreadPool(size_t n, size_t mqr = 0)
      : shutdown_(false), max_queued_requests_(mqr) {
    while (n) {
      threads_.emplace_back(worker(*this));
      n--;
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ~ThreadPool() override = default;

  bool enqueue(std::function<void()> fn) override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (max_queued_requests_ > 0 && jobs_.size() >= max_queued_requests_) {
        return false;
      }
      jobs_.push_back(std::move(fn));
    }

    cond_.notify_one();
    return true;
  }

  void shutdown() override {
    // Stop all worker threads...
    {
      std::unique_lock<std::mutex> lock(mutex_);
      shutdown_ = true;
    }

    cond_.notify_all();

    // Join...
    for (auto &t : threads_) {
      t.join();
    }
  }

private:
  struct worker {
    explicit worker(ThreadPool &pool) : pool_(pool) {}

    void operator()() {
      for (;;) {
        std::function<void()> fn;
        {
          std::unique_lock<std::mutex> lock(pool_.mutex_);

          pool_.cond_.wait(
              lock, [&] { return !pool_.jobs_.empty() || pool_.shutdown_; });

          if (pool_.shutdown_ && pool_.jobs_.empty()) { break; }

          fn = pool_.jobs_.front();
          pool_.jobs_.pop_front();
        }

        assert(true == static_cast<bool>(fn));
        fn();
      }

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT) && !defined(OPENSSL_IS_BORINGSSL) &&   \
    !defined(LIBRESSL_VERSION_NUMBER)
      OPENSSL_thread_stop();
#endif
    }

    ThreadPool &pool_;
  };
  friend struct worker;

  std::vector<std::thread> threads_;
  std::list<std::function<void()>> jobs_;

  bool shutdown_;
  size_t max_queued_requests_ = 0;

  std::condition_variable cond_;
  std::mutex mutex_;
};

using Logger = std::function<void(const Request &, const Response &)>;

using SocketOptions = std::function<void(socket_t sock)>;

void default_socket_options(socket_t sock);

const char *status_message(int status);

std::string get_bearer_token_auth(const Request &req);

namespace detail {

class MatcherBase {
public:
  virtual ~MatcherBase() = default;

  // Match request path and populate its matches and
  virtual bool match(Request &request) const = 0;
};

/**
 * Captures parameters in request path and stores them in Request::path_params
 *
 * Capture name is a substring of a pattern from : to /.
 * The rest of the pattern is matched agains the request path directly
 * Parameters are captured starting from the next character after
 * the end of the last matched static pattern fragment until the next /.
 *
 * Example pattern:
 * "/path/fragments/:capture/more/fragments/:second_capture"
 * Static fragments:
 * "/path/fragments/", "more/fragments/"
 *
 * Given the following request path:
 * "/path/fragments/:1/more/fragments/:2"
 * the resulting capture will be
 * {{"capture", "1"}, {"second_capture", "2"}}
 */
class PathParamsMatcher final : public MatcherBase {
public:
  PathParamsMatcher(const std::string &pattern);

  bool match(Request &request) const override;

private:
  // Treat segment separators as the end of path parameter capture
  // Does not need to handle query parameters as they are parsed before path
  // matching
  static constexpr char separator = '/';

  // Contains static path fragments to match against, excluding the '/' after
  // path params
  // Fragments are separated by path params
  std::vector<std::string> static_fragments_;
  // Stores the names of the path parameters to be used as keys in the
  // Request::path_params map
  std::vector<std::string> param_names_;
};

/**
 * Performs std::regex_match on request path
 * and stores the result in Request::matches
 *
 * Note that regex match is performed directly on the whole request.
 * This means that wildcard patterns may match multiple path segments with /:
 * "/begin/(.*)/end" will match both "/begin/middle/end" and "/begin/1/2/end".
 */
class RegexMatcher final : public MatcherBase {
public:
  RegexMatcher(const std::string &pattern) : regex_(pattern) {}

  bool match(Request &request) const override;

private:
  std::regex regex_;
};

ssize_t write_headers(Stream &strm, const Headers &headers);

} // namespace detail

class Server {
public:
  using Handler = std::function<void(const Request &, Response &)>;

  using ExceptionHandler =
      std::function<void(const Request &, Response &, std::exception_ptr ep)>;

  enum class HandlerResponse {
    Handled,
    Unhandled,
  };
  using HandlerWithResponse =
      std::function<HandlerResponse(const Request &, Response &)>;

  using HandlerWithContentReader = std::function<void(
      const Request &, Response &, const ContentReader &content_reader)>;

  using Expect100ContinueHandler =
      std::function<int(const Request &, Response &)>;

  Server();

  virtual ~Server();

  virtual bool is_valid() const;

  Server &Get(const std::string &pattern, Handler handler);
  Server &Post(const std::string &pattern, Handler handler);
  Server &Post(const std::string &pattern, HandlerWithContentReader handler);
  Server &Put(const std::string &pattern, Handler handler);
  Server &Put(const std::string &pattern, HandlerWithContentReader handler);
  Server &Patch(const std::string &pattern, Handler handler);
  Server &Patch(const std::string &pattern, HandlerWithContentReader handler);
  Server &Delete(const std::string &pattern, Handler handler);
  Server &Delete(const std::string &pattern, HandlerWithContentReader handler);
  Server &Options(const std::string &pattern, Handler handler);

  bool set_base_dir(const std::string &dir,
                    const std::string &mount_point = std::string());
  bool set_mount_point(const std::string &mount_point, const std::string &dir,
                       Headers headers = Headers());
  bool remove_mount_point(const std::string &mount_point);
  Server &set_file_extension_and_mimetype_mapping(const std::string &ext,
                                                  const std::string &mime);
  Server &set_default_file_mimetype(const std::string &mime);
  Server &set_file_request_handler(Handler handler);

  template <class ErrorHandlerFunc>
  Server &set_error_handler(ErrorHandlerFunc &&handler) {
    return set_error_handler_core(
        std::forward<ErrorHandlerFunc>(handler),
        std::is_convertible<ErrorHandlerFunc, HandlerWithResponse>{});
  }

  Server &set_exception_handler(ExceptionHandler handler);
  Server &set_pre_routing_handler(HandlerWithResponse handler);
  Server &set_post_routing_handler(Handler handler);

  Server &set_expect_100_continue_handler(Expect100ContinueHandler handler);
  Server &set_logger(Logger logger);

  Server &set_address_family(int family);
  Server &set_tcp_nodelay(bool on);
  Server &set_ipv6_v6only(bool on);
  Server &set_socket_options(SocketOptions socket_options);

  Server &set_default_headers(Headers headers);
  Server &
  set_header_writer(std::function<ssize_t(Stream &, Headers &)> const &writer);

  Server &set_keep_alive_max_count(size_t count);
  Server &set_keep_alive_timeout(time_t sec);

  Server &set_read_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  Server &set_read_timeout(const std::chrono::duration<Rep, Period> &duration);

  Server &set_write_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  Server &set_write_timeout(const std::chrono::duration<Rep, Period> &duration);

  Server &set_idle_interval(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  Server &set_idle_interval(const std::chrono::duration<Rep, Period> &duration);

  Server &set_payload_max_length(size_t length);

  bool bind_to_port(const std::string &host, int port, int socket_flags = 0);
  int bind_to_any_port(const std::string &host, int socket_flags = 0);
  bool listen_after_bind();

  bool listen(const std::string &host, int port, int socket_flags = 0);

  bool is_running() const;
  void wait_until_ready() const;
  void stop();
  void decommission();

  std::function<TaskQueue *(void)> new_task_queue;

protected:
  bool process_request(Stream &strm, const std::string &remote_addr,
                       int remote_port, const std::string &local_addr,
                       int local_port, bool close_connection,
                       bool &connection_closed,
                       const std::function<void(Request &)> &setup_request);

  std::atomic<socket_t> svr_sock_{INVALID_SOCKET};
  size_t keep_alive_max_count_ = CPPHTTPLIB_KEEPALIVE_MAX_COUNT;
  time_t keep_alive_timeout_sec_ = CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND;
  time_t read_timeout_sec_ = CPPHTTPLIB_SERVER_READ_TIMEOUT_SECOND;
  time_t read_timeout_usec_ = CPPHTTPLIB_SERVER_READ_TIMEOUT_USECOND;
  time_t write_timeout_sec_ = CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND;
  time_t write_timeout_usec_ = CPPHTTPLIB_SERVER_WRITE_TIMEOUT_USECOND;
  time_t idle_interval_sec_ = CPPHTTPLIB_IDLE_INTERVAL_SECOND;
  time_t idle_interval_usec_ = CPPHTTPLIB_IDLE_INTERVAL_USECOND;
  size_t payload_max_length_ = CPPHTTPLIB_PAYLOAD_MAX_LENGTH;

private:
  using Handlers =
      std::vector<std::pair<std::unique_ptr<detail::MatcherBase>, Handler>>;
  using HandlersForContentReader =
      std::vector<std::pair<std::unique_ptr<detail::MatcherBase>,
                            HandlerWithContentReader>>;

  static std::unique_ptr<detail::MatcherBase>
  make_matcher(const std::string &pattern);

  Server &set_error_handler_core(HandlerWithResponse handler, std::true_type);
  Server &set_error_handler_core(Handler handler, std::false_type);

  socket_t create_server_socket(const std::string &host, int port,
                                int socket_flags,
                                SocketOptions socket_options) const;
  int bind_internal(const std::string &host, int port, int socket_flags);
  bool listen_internal();

  bool routing(Request &req, Response &res, Stream &strm);
  bool handle_file_request(const Request &req, Response &res,
                           bool head = false);
  bool dispatch_request(Request &req, Response &res,
                        const Handlers &handlers) const;
  bool dispatch_request_for_content_reader(
      Request &req, Response &res, ContentReader content_reader,
      const HandlersForContentReader &handlers) const;

  bool parse_request_line(const char *s, Request &req) const;
  void apply_ranges(const Request &req, Response &res,
                    std::string &content_type, std::string &boundary) const;
  bool write_response(Stream &strm, bool close_connection, Request &req,
                      Response &res);
  bool write_response_with_content(Stream &strm, bool close_connection,
                                   const Request &req, Response &res);
  bool write_response_core(Stream &strm, bool close_connection,
                           const Request &req, Response &res,
                           bool need_apply_ranges);
  bool write_content_with_provider(Stream &strm, const Request &req,
                                   Response &res, const std::string &boundary,
                                   const std::string &content_type);
  bool read_content(Stream &strm, Request &req, Response &res);
  bool
  read_content_with_content_receiver(Stream &strm, Request &req, Response &res,
                                     ContentReceiver receiver,
                                     MultipartContentHeader multipart_header,
                                     ContentReceiver multipart_receiver);
  bool read_content_core(Stream &strm, Request &req, Response &res,
                         ContentReceiver receiver,
                         MultipartContentHeader multipart_header,
                         ContentReceiver multipart_receiver) const;

  virtual bool process_and_close_socket(socket_t sock);

  std::atomic<bool> is_running_{false};
  std::atomic<bool> is_decommisioned{false};

  struct MountPointEntry {
    std::string mount_point;
    std::string base_dir;
    Headers headers;
  };
  std::vector<MountPointEntry> base_dirs_;
  std::map<std::string, std::string> file_extension_and_mimetype_map_;
  std::string default_file_mimetype_ = "application/octet-stream";
  Handler file_request_handler_;

  Handlers get_handlers_;
  Handlers post_handlers_;
  HandlersForContentReader post_handlers_for_content_reader_;
  Handlers put_handlers_;
  HandlersForContentReader put_handlers_for_content_reader_;
  Handlers patch_handlers_;
  HandlersForContentReader patch_handlers_for_content_reader_;
  Handlers delete_handlers_;
  HandlersForContentReader delete_handlers_for_content_reader_;
  Handlers options_handlers_;

  HandlerWithResponse error_handler_;
  ExceptionHandler exception_handler_;
  HandlerWithResponse pre_routing_handler_;
  Handler post_routing_handler_;
  Expect100ContinueHandler expect_100_continue_handler_;

  Logger logger_;

  int address_family_ = AF_UNSPEC;
  bool tcp_nodelay_ = CPPHTTPLIB_TCP_NODELAY;
  bool ipv6_v6only_ = CPPHTTPLIB_IPV6_V6ONLY;
  SocketOptions socket_options_ = default_socket_options;

  Headers default_headers_;
  std::function<ssize_t(Stream &, Headers &)> header_writer_ =
      detail::write_headers;
};

enum class Error {
  Success = 0,
  Unknown,
  Connection,
  BindIPAddress,
  Read,
  Write,
  ExceedRedirectCount,
  Canceled,
  SSLConnection,
  SSLLoadingCerts,
  SSLServerVerification,
  SSLServerHostnameVerification,
  UnsupportedMultipartBoundaryChars,
  Compression,
  ConnectionTimeout,
  ProxyConnection,

  // For internal use only
  SSLPeerCouldBeClosed_,
};

std::string to_string(Error error);

std::ostream &operator<<(std::ostream &os, const Error &obj);

class Result {
public:
  Result() = default;
  Result(std::unique_ptr<Response> &&res, Error err,
         Headers &&request_headers = Headers{})
      : res_(std::move(res)), err_(err),
        request_headers_(std::move(request_headers)) {}
  // Response
  operator bool() const { return res_ != nullptr; }
  bool operator==(std::nullptr_t) const { return res_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return res_ != nullptr; }
  const Response &value() const { return *res_; }
  Response &value() { return *res_; }
  const Response &operator*() const { return *res_; }
  Response &operator*() { return *res_; }
  const Response *operator->() const { return res_.get(); }
  Response *operator->() { return res_.get(); }

  // Error
  Error error() const { return err_; }

  // Request Headers
  bool has_request_header(const std::string &key) const;
  std::string get_request_header_value(const std::string &key,
                                       const char *def = "",
                                       size_t id = 0) const;
  uint64_t get_request_header_value_u64(const std::string &key,
                                        uint64_t def = 0, size_t id = 0) const;
  size_t get_request_header_value_count(const std::string &key) const;

private:
  std::unique_ptr<Response> res_;
  Error err_ = Error::Unknown;
  Headers request_headers_;
};

class ClientImpl {
public:
  explicit ClientImpl(const std::string &host);

  explicit ClientImpl(const std::string &host, int port);

  explicit ClientImpl(const std::string &host, int port,
                      const std::string &client_cert_path,
                      const std::string &client_key_path);

  virtual ~ClientImpl();

  virtual bool is_valid() const;

  Result Get(const std::string &path);
  Result Get(const std::string &path, const Headers &headers);
  Result Get(const std::string &path, Progress progress);
  Result Get(const std::string &path, const Headers &headers,
             Progress progress);
  Result Get(const std::string &path, ContentReceiver content_receiver);
  Result Get(const std::string &path, const Headers &headers,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, ContentReceiver content_receiver,
             Progress progress);
  Result Get(const std::string &path, const Headers &headers,
             ContentReceiver content_receiver, Progress progress);
  Result Get(const std::string &path, ResponseHandler response_handler,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, const Headers &headers,
             ResponseHandler response_handler,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, ResponseHandler response_handler,
             ContentReceiver content_receiver, Progress progress);
  Result Get(const std::string &path, const Headers &headers,
             ResponseHandler response_handler, ContentReceiver content_receiver,
             Progress progress);

  Result Get(const std::string &path, const Params &params,
             const Headers &headers, Progress progress = nullptr);
  Result Get(const std::string &path, const Params &params,
             const Headers &headers, ContentReceiver content_receiver,
             Progress progress = nullptr);
  Result Get(const std::string &path, const Params &params,
             const Headers &headers, ResponseHandler response_handler,
             ContentReceiver content_receiver, Progress progress = nullptr);

  Result Head(const std::string &path);
  Result Head(const std::string &path, const Headers &headers);

  Result Post(const std::string &path);
  Result Post(const std::string &path, const Headers &headers);
  Result Post(const std::string &path, const char *body, size_t content_length,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers, const char *body,
              size_t content_length, const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers, const char *body,
              size_t content_length, const std::string &content_type,
              Progress progress);
  Result Post(const std::string &path, const std::string &body,
              const std::string &content_type);
  Result Post(const std::string &path, const std::string &body,
              const std::string &content_type, Progress progress);
  Result Post(const std::string &path, const Headers &headers,
              const std::string &body, const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              const std::string &body, const std::string &content_type,
              Progress progress);
  Result Post(const std::string &path, size_t content_length,
              ContentProvider content_provider,
              const std::string &content_type);
  Result Post(const std::string &path,
              ContentProviderWithoutLength content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              size_t content_length, ContentProvider content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              ContentProviderWithoutLength content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Params &params);
  Result Post(const std::string &path, const Headers &headers,
              const Params &params);
  Result Post(const std::string &path, const Headers &headers,
              const Params &params, Progress progress);
  Result Post(const std::string &path, const MultipartFormDataItems &items);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items, const std::string &boundary);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items,
              const MultipartFormDataProviderItems &provider_items);

  Result Put(const std::string &path);
  Result Put(const std::string &path, const char *body, size_t content_length,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers, const char *body,
             size_t content_length, const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers, const char *body,
             size_t content_length, const std::string &content_type,
             Progress progress);
  Result Put(const std::string &path, const std::string &body,
             const std::string &content_type);
  Result Put(const std::string &path, const std::string &body,
             const std::string &content_type, Progress progress);
  Result Put(const std::string &path, const Headers &headers,
             const std::string &body, const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             const std::string &body, const std::string &content_type,
             Progress progress);
  Result Put(const std::string &path, size_t content_length,
             ContentProvider content_provider, const std::string &content_type);
  Result Put(const std::string &path,
             ContentProviderWithoutLength content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             size_t content_length, ContentProvider content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             ContentProviderWithoutLength content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Params &params);
  Result Put(const std::string &path, const Headers &headers,
             const Params &params);
  Result Put(const std::string &path, const Headers &headers,
             const Params &params, Progress progress);
  Result Put(const std::string &path, const MultipartFormDataItems &items);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items, const std::string &boundary);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items,
             const MultipartFormDataProviderItems &provider_items);

  Result Patch(const std::string &path);
  Result Patch(const std::string &path, const char *body, size_t content_length,
               const std::string &content_type);
  Result Patch(const std::string &path, const char *body, size_t content_length,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const Headers &headers,
               const char *body, size_t content_length,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               const char *body, size_t content_length,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const std::string &body,
               const std::string &content_type);
  Result Patch(const std::string &path, const std::string &body,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const Headers &headers,
               const std::string &body, const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               const std::string &body, const std::string &content_type,
               Progress progress);
  Result Patch(const std::string &path, size_t content_length,
               ContentProvider content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path,
               ContentProviderWithoutLength content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               size_t content_length, ContentProvider content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               ContentProviderWithoutLength content_provider,
               const std::string &content_type);

  Result Delete(const std::string &path);
  Result Delete(const std::string &path, const Headers &headers);
  Result Delete(const std::string &path, const char *body,
                size_t content_length, const std::string &content_type);
  Result Delete(const std::string &path, const char *body,
                size_t content_length, const std::string &content_type,
                Progress progress);
  Result Delete(const std::string &path, const Headers &headers,
                const char *body, size_t content_length,
                const std::string &content_type);
  Result Delete(const std::string &path, const Headers &headers,
                const char *body, size_t content_length,
                const std::string &content_type, Progress progress);
  Result Delete(const std::string &path, const std::string &body,
                const std::string &content_type);
  Result Delete(const std::string &path, const std::string &body,
                const std::string &content_type, Progress progress);
  Result Delete(const std::string &path, const Headers &headers,
                const std::string &body, const std::string &content_type);
  Result Delete(const std::string &path, const Headers &headers,
                const std::string &body, const std::string &content_type,
                Progress progress);

  Result Options(const std::string &path);
  Result Options(const std::string &path, const Headers &headers);

  bool send(Request &req, Response &res, Error &error);
  Result send(const Request &req);

  void stop();

  std::string host() const;
  int port() const;

  size_t is_socket_open() const;
  socket_t socket() const;

  void set_hostname_addr_map(std::map<std::string, std::string> addr_map);

  void set_default_headers(Headers headers);

  void
  set_header_writer(std::function<ssize_t(Stream &, Headers &)> const &writer);

  void set_address_family(int family);
  void set_tcp_nodelay(bool on);
  void set_ipv6_v6only(bool on);
  void set_socket_options(SocketOptions socket_options);

  void set_connection_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void
  set_connection_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_read_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void set_read_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_write_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void set_write_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_basic_auth(const std::string &username, const std::string &password);
  void set_bearer_token_auth(const std::string &token);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_digest_auth(const std::string &username,
                       const std::string &password);
#endif

  void set_keep_alive(bool on);
  void set_follow_location(bool on);

  void set_url_encode(bool on);

  void set_compress(bool on);

  void set_decompress(bool on);

  void set_interface(const std::string &intf);

  void set_proxy(const std::string &host, int port);
  void set_proxy_basic_auth(const std::string &username,
                            const std::string &password);
  void set_proxy_bearer_token_auth(const std::string &token);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_proxy_digest_auth(const std::string &username,
                             const std::string &password);
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_ca_cert_path(const std::string &ca_cert_file_path,
                        const std::string &ca_cert_dir_path = std::string());
  void set_ca_cert_store(X509_STORE *ca_cert_store);
  X509_STORE *create_ca_cert_store(const char *ca_cert, std::size_t size) const;
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void enable_server_certificate_verification(bool enabled);
  void enable_server_hostname_verification(bool enabled);
  void set_server_certificate_verifier(std::function<bool(SSL *ssl)> verifier);
#endif

  void set_logger(Logger logger);

protected:
  struct Socket {
    socket_t sock = INVALID_SOCKET;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    SSL *ssl = nullptr;
#endif

    bool is_open() const { return sock != INVALID_SOCKET; }
  };

  virtual bool create_and_connect_socket(Socket &socket, Error &error);

  // All of:
  //   shutdown_ssl
  //   shutdown_socket
  //   close_socket
  // should ONLY be called when socket_mutex_ is locked.
  // Also, shutdown_ssl and close_socket should also NOT be called concurrently
  // with a DIFFERENT thread sending requests using that socket.
  virtual void shutdown_ssl(Socket &socket, bool shutdown_gracefully);
  void shutdown_socket(Socket &socket) const;
  void close_socket(Socket &socket);

  bool process_request(Stream &strm, Request &req, Response &res,
                       bool close_connection, Error &error);

  bool write_content_with_provider(Stream &strm, const Request &req,
                                   Error &error) const;

  void copy_settings(const ClientImpl &rhs);

  // Socket endpoint information
  const std::string host_;
  const int port_;
  const std::string host_and_port_;

  // Current open socket
  Socket socket_;
  mutable std::mutex socket_mutex_;
  std::recursive_mutex request_mutex_;

  // These are all protected under socket_mutex
  size_t socket_requests_in_flight_ = 0;
  std::thread::id socket_requests_are_from_thread_ = std::thread::id();
  bool socket_should_be_closed_when_request_is_done_ = false;

  // Hostname-IP map
  std::map<std::string, std::string> addr_map_;

  // Default headers
  Headers default_headers_;

  // Header writer
  std::function<ssize_t(Stream &, Headers &)> header_writer_ =
      detail::write_headers;

  // Settings
  std::string client_cert_path_;
  std::string client_key_path_;

  time_t connection_timeout_sec_ = CPPHTTPLIB_CONNECTION_TIMEOUT_SECOND;
  time_t connection_timeout_usec_ = CPPHTTPLIB_CONNECTION_TIMEOUT_USECOND;
  time_t read_timeout_sec_ = CPPHTTPLIB_CLIENT_READ_TIMEOUT_SECOND;
  time_t read_timeout_usec_ = CPPHTTPLIB_CLIENT_READ_TIMEOUT_USECOND;
  time_t write_timeout_sec_ = CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_SECOND;
  time_t write_timeout_usec_ = CPPHTTPLIB_CLIENT_WRITE_TIMEOUT_USECOND;

  std::string basic_auth_username_;
  std::string basic_auth_password_;
  std::string bearer_token_auth_token_;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  std::string digest_auth_username_;
  std::string digest_auth_password_;
#endif

  bool keep_alive_ = false;
  bool follow_location_ = false;

  bool url_encode_ = true;

  int address_family_ = AF_UNSPEC;
  bool tcp_nodelay_ = CPPHTTPLIB_TCP_NODELAY;
  bool ipv6_v6only_ = CPPHTTPLIB_IPV6_V6ONLY;
  SocketOptions socket_options_ = nullptr;

  bool compress_ = false;
  bool decompress_ = true;

  std::string interface_;

  std::string proxy_host_;
  int proxy_port_ = -1;

  std::string proxy_basic_auth_username_;
  std::string proxy_basic_auth_password_;
  std::string proxy_bearer_token_auth_token_;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  std::string proxy_digest_auth_username_;
  std::string proxy_digest_auth_password_;
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  std::string ca_cert_file_path_;
  std::string ca_cert_dir_path_;

  X509_STORE *ca_cert_store_ = nullptr;
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  bool server_certificate_verification_ = true;
  bool server_hostname_verification_ = true;
  std::function<bool(SSL *ssl)> server_certificate_verifier_;
#endif

  Logger logger_;

private:
  bool send_(Request &req, Response &res, Error &error);
  Result send_(Request &&req);

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  bool is_ssl_peer_could_be_closed(SSL *ssl) const;
#endif
  socket_t create_client_socket(Error &error) const;
  bool read_response_line(Stream &strm, const Request &req,
                          Response &res) const;
  bool write_request(Stream &strm, Request &req, bool close_connection,
                     Error &error);
  bool redirect(Request &req, Response &res, Error &error);
  bool handle_request(Stream &strm, Request &req, Response &res,
                      bool close_connection, Error &error);
  std::unique_ptr<Response> send_with_content_provider(
      Request &req, const char *body, size_t content_length,
      ContentProvider content_provider,
      ContentProviderWithoutLength content_provider_without_length,
      const std::string &content_type, Error &error);
  Result send_with_content_provider(
      const std::string &method, const std::string &path,
      const Headers &headers, const char *body, size_t content_length,
      ContentProvider content_provider,
      ContentProviderWithoutLength content_provider_without_length,
      const std::string &content_type, Progress progress);
  ContentProviderWithoutLength get_multipart_content_provider(
      const std::string &boundary, const MultipartFormDataItems &items,
      const MultipartFormDataProviderItems &provider_items) const;

  std::string adjust_host_string(const std::string &host) const;

  virtual bool process_socket(const Socket &socket,
                              std::function<bool(Stream &strm)> callback);
  virtual bool is_ssl() const;
};

class Client {
public:
  // Universal interface
  explicit Client(const std::string &scheme_host_port);

  explicit Client(const std::string &scheme_host_port,
                  const std::string &client_cert_path,
                  const std::string &client_key_path);

  // HTTP only interface
  explicit Client(const std::string &host, int port);

  explicit Client(const std::string &host, int port,
                  const std::string &client_cert_path,
                  const std::string &client_key_path);

  Client(Client &&) = default;
  Client &operator=(Client &&) = default;

  ~Client();

  bool is_valid() const;

  Result Get(const std::string &path);
  Result Get(const std::string &path, const Headers &headers);
  Result Get(const std::string &path, Progress progress);
  Result Get(const std::string &path, const Headers &headers,
             Progress progress);
  Result Get(const std::string &path, ContentReceiver content_receiver);
  Result Get(const std::string &path, const Headers &headers,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, ContentReceiver content_receiver,
             Progress progress);
  Result Get(const std::string &path, const Headers &headers,
             ContentReceiver content_receiver, Progress progress);
  Result Get(const std::string &path, ResponseHandler response_handler,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, const Headers &headers,
             ResponseHandler response_handler,
             ContentReceiver content_receiver);
  Result Get(const std::string &path, const Headers &headers,
             ResponseHandler response_handler, ContentReceiver content_receiver,
             Progress progress);
  Result Get(const std::string &path, ResponseHandler response_handler,
             ContentReceiver content_receiver, Progress progress);

  Result Get(const std::string &path, const Params &params,
             const Headers &headers, Progress progress = nullptr);
  Result Get(const std::string &path, const Params &params,
             const Headers &headers, ContentReceiver content_receiver,
             Progress progress = nullptr);
  Result Get(const std::string &path, const Params &params,
             const Headers &headers, ResponseHandler response_handler,
             ContentReceiver content_receiver, Progress progress = nullptr);

  Result Head(const std::string &path);
  Result Head(const std::string &path, const Headers &headers);

  Result Post(const std::string &path);
  Result Post(const std::string &path, const Headers &headers);
  Result Post(const std::string &path, const char *body, size_t content_length,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers, const char *body,
              size_t content_length, const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers, const char *body,
              size_t content_length, const std::string &content_type,
              Progress progress);
  Result Post(const std::string &path, const std::string &body,
              const std::string &content_type);
  Result Post(const std::string &path, const std::string &body,
              const std::string &content_type, Progress progress);
  Result Post(const std::string &path, const Headers &headers,
              const std::string &body, const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              const std::string &body, const std::string &content_type,
              Progress progress);
  Result Post(const std::string &path, size_t content_length,
              ContentProvider content_provider,
              const std::string &content_type);
  Result Post(const std::string &path,
              ContentProviderWithoutLength content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              size_t content_length, ContentProvider content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Headers &headers,
              ContentProviderWithoutLength content_provider,
              const std::string &content_type);
  Result Post(const std::string &path, const Params &params);
  Result Post(const std::string &path, const Headers &headers,
              const Params &params);
  Result Post(const std::string &path, const Headers &headers,
              const Params &params, Progress progress);
  Result Post(const std::string &path, const MultipartFormDataItems &items);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items, const std::string &boundary);
  Result Post(const std::string &path, const Headers &headers,
              const MultipartFormDataItems &items,
              const MultipartFormDataProviderItems &provider_items);

  Result Put(const std::string &path);
  Result Put(const std::string &path, const char *body, size_t content_length,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers, const char *body,
             size_t content_length, const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers, const char *body,
             size_t content_length, const std::string &content_type,
             Progress progress);
  Result Put(const std::string &path, const std::string &body,
             const std::string &content_type);
  Result Put(const std::string &path, const std::string &body,
             const std::string &content_type, Progress progress);
  Result Put(const std::string &path, const Headers &headers,
             const std::string &body, const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             const std::string &body, const std::string &content_type,
             Progress progress);
  Result Put(const std::string &path, size_t content_length,
             ContentProvider content_provider, const std::string &content_type);
  Result Put(const std::string &path,
             ContentProviderWithoutLength content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             size_t content_length, ContentProvider content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Headers &headers,
             ContentProviderWithoutLength content_provider,
             const std::string &content_type);
  Result Put(const std::string &path, const Params &params);
  Result Put(const std::string &path, const Headers &headers,
             const Params &params);
  Result Put(const std::string &path, const Headers &headers,
             const Params &params, Progress progress);
  Result Put(const std::string &path, const MultipartFormDataItems &items);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items, const std::string &boundary);
  Result Put(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items,
             const MultipartFormDataProviderItems &provider_items);

  Result Patch(const std::string &path);
  Result Patch(const std::string &path, const char *body, size_t content_length,
               const std::string &content_type);
  Result Patch(const std::string &path, const char *body, size_t content_length,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const Headers &headers,
               const char *body, size_t content_length,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               const char *body, size_t content_length,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const std::string &body,
               const std::string &content_type);
  Result Patch(const std::string &path, const std::string &body,
               const std::string &content_type, Progress progress);
  Result Patch(const std::string &path, const Headers &headers,
               const std::string &body, const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               const std::string &body, const std::string &content_type,
               Progress progress);
  Result Patch(const std::string &path, size_t content_length,
               ContentProvider content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path,
               ContentProviderWithoutLength content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               size_t content_length, ContentProvider content_provider,
               const std::string &content_type);
  Result Patch(const std::string &path, const Headers &headers,
               ContentProviderWithoutLength content_provider,
               const std::string &content_type);

  Result Delete(const std::string &path);
  Result Delete(const std::string &path, const Headers &headers);
  Result Delete(const std::string &path, const char *body,
                size_t content_length, const std::string &content_type);
  Result Delete(const std::string &path, const char *body,
                size_t content_length, const std::string &content_type,
                Progress progress);
  Result Delete(const std::string &path, const Headers &headers,
                const char *body, size_t content_length,
                const std::string &content_type);
  Result Delete(const std::string &path, const Headers &headers,
                const char *body, size_t content_length,
                const std::string &content_type, Progress progress);
  Result Delete(const std::string &path, const std::string &body,
                const std::string &content_type);
  Result Delete(const std::string &path, const std::string &body,
                const std::string &content_type, Progress progress);
  Result Delete(const std::string &path, const Headers &headers,
                const std::string &body, const std::string &content_type);
  Result Delete(const std::string &path, const Headers &headers,
                const std::string &body, const std::string &content_type,
                Progress progress);

  Result Options(const std::string &path);
  Result Options(const std::string &path, const Headers &headers);

  bool send(Request &req, Response &res, Error &error);
  Result send(const Request &req);

  void stop();

  std::string host() const;
  int port() const;

  size_t is_socket_open() const;
  socket_t socket() const;

  void set_hostname_addr_map(std::map<std::string, std::string> addr_map);

  void set_default_headers(Headers headers);

  void
  set_header_writer(std::function<ssize_t(Stream &, Headers &)> const &writer);

  void set_address_family(int family);
  void set_tcp_nodelay(bool on);
  void set_socket_options(SocketOptions socket_options);

  void set_connection_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void
  set_connection_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_read_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void set_read_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_write_timeout(time_t sec, time_t usec = 0);
  template <class Rep, class Period>
  void set_write_timeout(const std::chrono::duration<Rep, Period> &duration);

  void set_basic_auth(const std::string &username, const std::string &password);
  void set_bearer_token_auth(const std::string &token);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_digest_auth(const std::string &username,
                       const std::string &password);
#endif

  void set_keep_alive(bool on);
  void set_follow_location(bool on);

  void set_url_encode(bool on);

  void set_compress(bool on);

  void set_decompress(bool on);

  void set_interface(const std::string &intf);

  void set_proxy(const std::string &host, int port);
  void set_proxy_basic_auth(const std::string &username,
                            const std::string &password);
  void set_proxy_bearer_token_auth(const std::string &token);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_proxy_digest_auth(const std::string &username,
                             const std::string &password);
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void enable_server_certificate_verification(bool enabled);
  void enable_server_hostname_verification(bool enabled);
  void set_server_certificate_verifier(std::function<bool(SSL *ssl)> verifier);
#endif

  void set_logger(Logger logger);

  // SSL
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  void set_ca_cert_path(const std::string &ca_cert_file_path,
                        const std::string &ca_cert_dir_path = std::string());

  void set_ca_cert_store(X509_STORE *ca_cert_store);
  void load_ca_cert_store(const char *ca_cert, std::size_t size);

  long get_openssl_verify_result() const;

  SSL_CTX *ssl_context() const;
#endif

private:
  std::unique_ptr<ClientImpl> cli_;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  bool is_ssl_ = false;
#endif
};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
class SSLServer : public Server {
public:
  SSLServer(const char *cert_path, const char *private_key_path,
            const char *client_ca_cert_file_path = nullptr,
            const char *client_ca_cert_dir_path = nullptr,
            const char *private_key_password = nullptr);

  SSLServer(X509 *cert, EVP_PKEY *private_key,
            X509_STORE *client_ca_cert_store = nullptr);

  SSLServer(
      const std::function<bool(SSL_CTX &ssl_ctx)> &setup_ssl_ctx_callback);

  ~SSLServer() override;

  bool is_valid() const override;

  SSL_CTX *ssl_context() const;

  void update_certs(X509 *cert, EVP_PKEY *private_key,
                    X509_STORE *client_ca_cert_store = nullptr);

private:
  bool process_and_close_socket(socket_t sock) override;

  SSL_CTX *ctx_;
  std::mutex ctx_mutex_;
};

class SSLClient final : public ClientImpl {
public:
  explicit SSLClient(const std::string &host);

  explicit SSLClient(const std::string &host, int port);

  explicit SSLClient(const std::string &host, int port,
                     const std::string &client_cert_path,
                     const std::string &client_key_path,
                     const std::string &private_key_password = std::string());

  explicit SSLClient(const std::string &host, int port, X509 *client_cert,
                     EVP_PKEY *client_key,
                     const std::string &private_key_password = std::string());

  ~SSLClient() override;

  bool is_valid() const override;

  void set_ca_cert_store(X509_STORE *ca_cert_store);
  void load_ca_cert_store(const char *ca_cert, std::size_t size);

  long get_openssl_verify_result() const;

  SSL_CTX *ssl_context() const;

private:
  bool create_and_connect_socket(Socket &socket, Error &error) override;
  void shutdown_ssl(Socket &socket, bool shutdown_gracefully) override;
  void shutdown_ssl_impl(Socket &socket, bool shutdown_gracefully);

  bool process_socket(const Socket &socket,
                      std::function<bool(Stream &strm)> callback) override;
  bool is_ssl() const override;

  bool connect_with_proxy(Socket &sock, Response &res, bool &success,
                          Error &error);
  bool initialize_ssl(Socket &socket, Error &error);

  bool load_certs();

  bool verify_host(X509 *server_cert) const;
  bool verify_host_with_subject_alt_name(X509 *server_cert) const;
  bool verify_host_with_common_name(X509 *server_cert) const;
  bool check_host_name(const char *pattern, size_t pattern_len) const;

  SSL_CTX *ctx_;
  std::mutex ctx_mutex_;
  std::once_flag initialize_cert_;

  std::vector<std::string> host_components_;

  long verify_result_ = 0;

  friend class ClientImpl;
};
#endif

/*
 * Implementation of template methods.
 */

namespace detail {

template <typename T, typename U>
inline void duration_to_sec_and_usec(const T &duration, U callback) {
  auto sec = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  auto usec = std::chrono::duration_cast<std::chrono::microseconds>(
                  duration - std::chrono::seconds(sec))
                  .count();
  callback(static_cast<time_t>(sec), static_cast<time_t>(usec));
}

inline uint64_t get_header_value_u64(const Headers &headers,
                                     const std::string &key, uint64_t def,
                                     size_t id) {
  auto rng = headers.equal_range(key);
  auto it = rng.first;
  std::advance(it, static_cast<ssize_t>(id));
  if (it != rng.second) {
    return std::strtoull(it->second.data(), nullptr, 10);
  }
  return def;
}

} // namespace detail

inline uint64_t Request::get_header_value_u64(const std::string &key,
                                              uint64_t def, size_t id) const {
  return detail::get_header_value_u64(headers, key, def, id);
}

inline uint64_t Response::get_header_value_u64(const std::string &key,
                                               uint64_t def, size_t id) const {
  return detail::get_header_value_u64(headers, key, def, id);
}

inline void default_socket_options(socket_t sock) {
  int opt = 1;
#ifdef _WIN32
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const char *>(&opt), sizeof(opt));
  setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
             reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
#ifdef SO_REUSEPORT
  setsockopt(sock, SOL_SOCKET, SO_REUSEPORT,
             reinterpret_cast<const void *>(&opt), sizeof(opt));
#else
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const void *>(&opt), sizeof(opt));
#endif
#endif
}

inline const char *status_message(int status) {
  switch (status) {
  case StatusCode::Continue_100: return "Continue";
  case StatusCode::SwitchingProtocol_101: return "Switching Protocol";
  case StatusCode::Processing_102: return "Processing";
  case StatusCode::EarlyHints_103: return "Early Hints";
  case StatusCode::OK_200: return "OK";
  case StatusCode::Created_201: return "Created";
  case StatusCode::Accepted_202: return "Accepted";
  case StatusCode::NonAuthoritativeInformation_203:
    return "Non-Authoritative Information";
  case StatusCode::NoContent_204: return "No Content";
  case StatusCode::ResetContent_205: return "Reset Content";
  case StatusCode::PartialContent_206: return "Partial Content";
  case StatusCode::MultiStatus_207: return "Multi-Status";
  case StatusCode::AlreadyReported_208: return "Already Reported";
  case StatusCode::IMUsed_226: return "IM Used";
  case StatusCode::MultipleChoices_300: return "Multiple Choices";
  case StatusCode::MovedPermanently_301: return "Moved Permanently";
  case StatusCode::Found_302: return "Found";
  case StatusCode::SeeOther_303: return "See Other";
  case StatusCode::NotModified_304: return "Not Modified";
  case StatusCode::UseProxy_305: return "Use Proxy";
  case StatusCode::unused_306: return "unused";
  case StatusCode::TemporaryRedirect_307: return "Temporary Redirect";
  case StatusCode::PermanentRedirect_308: return "Permanent Redirect";
  case StatusCode::BadRequest_400: return "Bad Request";
  case StatusCode::Unauthorized_401: return "Unauthorized";
  case StatusCode::PaymentRequired_402: return "Payment Required";
  case StatusCode::Forbidden_403: return "Forbidden";
  case StatusCode::NotFound_404: return "Not Found";
  case StatusCode::MethodNotAllowed_405: return "Method Not Allowed";
  case StatusCode::NotAcceptable_406: return "Not Acceptable";
  case StatusCode::ProxyAuthenticationRequired_407:
    return "Proxy Authentication Required";
  case StatusCode::RequestTimeout_408: return "Request Timeout";
  case StatusCode::Conflict_409: return "Conflict";
  case StatusCode::Gone_410: return "Gone";
  case StatusCode::LengthRequired_411: return "Length Required";
  case StatusCode::PreconditionFailed_412: return "Precondition Failed";
  case StatusCode::PayloadTooLarge_413: return "Payload Too Large";
  case StatusCode::UriTooLong_414: return "URI Too Long";
  case StatusCode::UnsupportedMediaType_415: return "Unsupported Media Type";
  case StatusCode::RangeNotSatisfiable_416: return "Range Not Satisfiable";
  case StatusCode::ExpectationFailed_417: return "Expectation Failed";
  case StatusCode::ImATeapot_418: return "I'm a teapot";
  case StatusCode::MisdirectedRequest_421: return "Misdirected Request";
  case StatusCode::UnprocessableContent_422: return "Unprocessable Content";
  case StatusCode::Locked_423: return "Locked";
  case StatusCode::FailedDependency_424: return "Failed Dependency";
  case StatusCode::TooEarly_425: return "Too Early";
  case StatusCode::UpgradeRequired_426: return "Upgrade Required";
  case StatusCode::PreconditionRequired_428: return "Precondition Required";
  case StatusCode::TooManyRequests_429: return "Too Many Requests";
  case StatusCode::RequestHeaderFieldsTooLarge_431:
    return "Request Header Fields Too Large";
  case StatusCode::UnavailableForLegalReasons_451:
    return "Unavailable For Legal Reasons";
  case StatusCode::NotImplemented_501: return "Not Implemented";
  case StatusCode::BadGateway_502: return "Bad Gateway";
  case StatusCode::ServiceUnavailable_503: return "Service Unavailable";
  case StatusCode::GatewayTimeout_504: return "Gateway Timeout";
  case StatusCode::HttpVersionNotSupported_505:
    return "HTTP Version Not Supported";
  case StatusCode::VariantAlsoNegotiates_506: return "Variant Also Negotiates";
  case StatusCode::InsufficientStorage_507: return "Insufficient Storage";
  case StatusCode::LoopDetected_508: return "Loop Detected";
  case StatusCode::NotExtended_510: return "Not Extended";
  case StatusCode::NetworkAuthenticationRequired_511:
    return "Network Authentication Required";

  default:
  case StatusCode::InternalServerError_500: return "Internal Server Error";
  }
}

inline std::string get_bearer_token_auth(const Request &req) {
  if (req.has_header("Authorization")) {
    static std::string BearerHeaderPrefix = "Bearer ";
    return req.get_header_value("Authorization")
        .substr(BearerHeaderPrefix.length());
  }
  return "";
}

template <class Rep, class Period>
inline Server &
Server::set_read_timeout(const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(
      duration, [&](time_t sec, time_t usec) { set_read_timeout(sec, usec); });
  return *this;
}

template <class Rep, class Period>
inline Server &
Server::set_write_timeout(const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(
      duration, [&](time_t sec, time_t usec) { set_write_timeout(sec, usec); });
  return *this;
}

template <class Rep, class Period>
inline Server &
Server::set_idle_interval(const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(
      duration, [&](time_t sec, time_t usec) { set_idle_interval(sec, usec); });
  return *this;
}

inline std::string to_string(const Error error) {
  switch (error) {
  case Error::Success: return "Success (no error)";
  case Error::Connection: return "Could not establish connection";
  case Error::BindIPAddress: return "Failed to bind IP address";
  case Error::Read: return "Failed to read connection";
  case Error::Write: return "Failed to write connection";
  case Error::ExceedRedirectCount: return "Maximum redirect count exceeded";
  case Error::Canceled: return "Connection handling canceled";
  case Error::SSLConnection: return "SSL connection failed";
  case Error::SSLLoadingCerts: return "SSL certificate loading failed";
  case Error::SSLServerVerification: return "SSL server verification failed";
  case Error::SSLServerHostnameVerification:
    return "SSL server hostname verification failed";
  case Error::UnsupportedMultipartBoundaryChars:
    return "Unsupported HTTP multipart boundary characters";
  case Error::Compression: return "Compression failed";
  case Error::ConnectionTimeout: return "Connection timed out";
  case Error::ProxyConnection: return "Proxy connection failed";
  case Error::Unknown: return "Unknown";
  default: break;
  }

  return "Invalid";
}

inline std::ostream &operator<<(std::ostream &os, const Error &obj) {
  os << to_string(obj);
  os << " (" << static_cast<std::underlying_type<Error>::type>(obj) << ')';
  return os;
}

inline uint64_t Result::get_request_header_value_u64(const std::string &key,
                                                     uint64_t def,
                                                     size_t id) const {
  return detail::get_header_value_u64(request_headers_, key, def, id);
}

template <class Rep, class Period>
inline void ClientImpl::set_connection_timeout(
    const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(duration, [&](time_t sec, time_t usec) {
    set_connection_timeout(sec, usec);
  });
}

template <class Rep, class Period>
inline void ClientImpl::set_read_timeout(
    const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(
      duration, [&](time_t sec, time_t usec) { set_read_timeout(sec, usec); });
}

template <class Rep, class Period>
inline void ClientImpl::set_write_timeout(
    const std::chrono::duration<Rep, Period> &duration) {
  detail::duration_to_sec_and_usec(
      duration, [&](time_t sec, time_t usec) { set_write_timeout(sec, usec); });
}

template <class Rep, class Period>
inline void Client::set_connection_timeout(
    const std::chrono::duration<Rep, Period> &duration) {
  cli_->set_connection_timeout(duration);
}

template <class Rep, class Period>
inline void
Client::set_read_timeout(const std::chrono::duration<Rep, Period> &duration) {
  cli_->set_read_timeout(duration);
}

template <class Rep, class Period>
inline void
Client::set_write_timeout(const std::chrono::duration<Rep, Period> &duration) {
  cli_->set_write_timeout(duration);
}

/*
 * Forward declarations and types that will be part of the .h file if split into
 * .h + .cc.
 */

std::string hosted_at(const std::string &hostname);

void hosted_at(const std::string &hostname, std::vector<std::string> &addrs);

std::string append_query_params(const std::string &path, const Params &params);

std::pair<std::string, std::string> make_range_header(const Ranges &ranges);

std::pair<std::string, std::string>
make_basic_authentication_header(const std::string &username,
                                 const std::string &password,
                                 bool is_proxy = false);

namespace detail {

#if defined(_WIN32)
inline std::wstring u8string_to_wstring(const char *s) {
  std::wstring ws;
  auto len = static_cast<int>(strlen(s));
  auto wlen = ::MultiByteToWideChar(CP_UTF8, 0, s, len, nullptr, 0);
  if (wlen > 0) {
    ws.resize(wlen);
    wlen = ::MultiByteToWideChar(
        CP_UTF8, 0, s, len,
        const_cast<LPWSTR>(reinterpret_cast<LPCWSTR>(ws.data())), wlen);
    if (wlen != static_cast<int>(ws.size())) { ws.clear(); }
  }
  return ws;
}
#endif

struct FileStat {
  FileStat(const std::string &path);
  bool is_file() const;
  bool is_dir() const;

private:
#if defined(_WIN32)
  struct _stat st_;
#else
  struct stat st_;
#endif
  int ret_ = -1;
};

std::string encode_query_param(const std::string &value);

std::string decode_url(const std::string &s, bool convert_plus_to_space);

void read_file(const std::string &path, std::string &out);

std::string trim_copy(const std::string &s);

void divide(
    const char *data, std::size_t size, char d,
    std::function<void(const char *, std::size_t, const char *, std::size_t)>
        fn);

void divide(
    const std::string &str, char d,
    std::function<void(const char *, std::size_t, const char *, std::size_t)>
        fn);

void split(const char *b, const char *e, char d,
           std::function<void(const char *, const char *)> fn);

void split(const char *b, const char *e, char d, size_t m,
           std::function<void(const char *, const char *)> fn);

bool process_client_socket(socket_t sock, time_t read_timeout_sec,
                           time_t read_timeout_usec, time_t write_timeout_sec,
                           time_t write_timeout_usec,
                           std::function<bool(Stream &)> callback);

socket_t create_client_socket(const std::string &host, const std::string &ip,
                              int port, int address_family, bool tcp_nodelay,
                              bool ipv6_v6only, SocketOptions socket_options,
                              time_t connection_timeout_sec,
                              time_t connection_timeout_usec,
                              time_t read_timeout_sec, time_t read_timeout_usec,
                              time_t write_timeout_sec,
                              time_t write_timeout_usec,
                              const std::string &intf, Error &error);

const char *get_header_value(const Headers &headers, const std::string &key,
                             const char *def, size_t id);

std::string params_to_query_str(const Params &params);

void parse_query_text(const char *data, std::size_t size, Params &params);

void parse_query_text(const std::string &s, Params &params);

bool parse_multipart_boundary(const std::string &content_type,
                              std::string &boundary);

bool parse_range_header(const std::string &s, Ranges &ranges);

int close_socket(socket_t sock);

ssize_t send_socket(socket_t sock, const void *ptr, size_t size, int flags);

ssize_t read_socket(socket_t sock, void *ptr, size_t size, int flags);

enum class EncodingType { None = 0, Gzip, Brotli };

EncodingType encoding_type(const Request &req, const Response &res);

class BufferStream final : public Stream {
public:
  BufferStream() = default;
  ~BufferStream() override = default;

  bool is_readable() const override;
  bool is_writable() const override;
  ssize_t read(char *ptr, size_t size) override;
  ssize_t write(const char *ptr, size_t size) override;
  void get_remote_ip_and_port(std::string &ip, int &port) const override;
  void get_local_ip_and_port(std::string &ip, int &port) const override;
  socket_t socket() const override;

  const std::string &get_buffer() const;

private:
  std::string buffer;
  size_t position = 0;
};

class compressor {
public:
  virtual ~compressor() = default;

  typedef std::function<bool(const char *data, size_t data_len)> Callback;
  virtual bool compress(const char *data, size_t data_length, bool last,
                        Callback callback) = 0;
};

class decompressor {
public:
  virtual ~decompressor() = default;

  virtual bool is_valid() const = 0;

  typedef std::function<bool(const char *data, size_t data_len)> Callback;
  virtual bool decompress(const char *data, size_t data_length,
                          Callback callback) = 0;
};

class nocompressor final : public compressor {
public:
  ~nocompressor() override = default;

  bool compress(const char *data, size_t data_length, bool /*last*/,
                Callback callback) override;
};

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
class gzip_compressor final : public compressor {
public:
  gzip_compressor();
  ~gzip_compressor() override;

  bool compress(const char *data, size_t data_length, bool last,
                Callback callback) override;

private:
  bool is_valid_ = false;
  z_stream strm_;
};

class gzip_decompressor final : public decompressor {
public:
  gzip_decompressor();
  ~gzip_decompressor() override;

  bool is_valid() const override;

  bool decompress(const char *data, size_t data_length,
                  Callback callback) override;

private:
  bool is_valid_ = false;
  z_stream strm_;
};
#endif

#ifdef CPPHTTPLIB_BROTLI_SUPPORT
class brotli_compressor final : public compressor {
public:
  brotli_compressor();
  ~brotli_compressor();

  bool compress(const char *data, size_t data_length, bool last,
                Callback callback) override;

private:
  BrotliEncoderState *state_ = nullptr;
};

class brotli_decompressor final : public decompressor {
public:
  brotli_decompressor();
  ~brotli_decompressor();

  bool is_valid() const override;

  bool decompress(const char *data, size_t data_length,
                  Callback callback) override;

private:
  BrotliDecoderResult decoder_r;
  BrotliDecoderState *decoder_s = nullptr;
};
#endif

// NOTE: until the read size reaches `fixed_buffer_size`, use `fixed_buffer`
// to store data. The call can set memory on stack for performance.
class stream_line_reader {
public:
  stream_line_reader(Stream &strm, char *fixed_buffer,
                     size_t fixed_buffer_size);
  const char *ptr() const;
  size_t size() const;
  bool end_with_crlf() const;
  bool getline();

private:
  void append(char c);

  Stream &strm_;
  char *fixed_buffer_;
  const size_t fixed_buffer_size_;
  size_t fixed_buffer_used_size_ = 0;
  std::string glowable_buffer_;
};

class mmap {
public:
  mmap(const char *path);
  ~mmap();

  bool open(const char *path);
  void close();

  bool is_open() const;
  size_t size() const;
  const char *data() const;

private:
#if defined(_WIN32)
  HANDLE hFile_ = NULL;
  HANDLE hMapping_ = NULL;
#else
  int fd_ = -1;
#endif
  size_t size_ = 0;
  void *addr_ = nullptr;
  bool is_open_empty_file = false;
};

} // namespace detail


} // namespace httplib

#if defined(_WIN32) && defined(CPPHTTPLIB_USE_POLL)
#undef poll
#endif

#endif // CPPHTTPLIB_HTTPLIB_H
