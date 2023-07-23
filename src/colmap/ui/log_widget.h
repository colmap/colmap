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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/controllers/option_manager.h"

#include <QtGui>
#include <QtWidgets>
#include <fstream>
#include <iostream>

namespace colmap {

template <class Elem = char, class Tr = std::char_traits<Elem>>
class StandardOutputRedirector : public std::basic_streambuf<Elem, Tr> {
  typedef void (*cb_func_ptr)(const Elem*, std::streamsize count, void* data);

 public:
  StandardOutputRedirector(std::ostream& stream,
                           cb_func_ptr cb_func,
                           void* data)
      : stream_(stream), cb_func_(cb_func), data_(data) {
    buf_ = stream_.rdbuf(this);
  };

  ~StandardOutputRedirector() { stream_.rdbuf(buf_); }

  std::streamsize xsputn(const Elem* ptr, std::streamsize count) {
    cb_func_(ptr, count, data_);
    return count;
  }

  typename Tr::int_type overflow(typename Tr::int_type v) {
    Elem ch = Tr::to_char_type(v);
    cb_func_(&ch, 1, data_);
    return Tr::not_eof(v);
  }

 private:
  std::basic_ostream<Elem, Tr>& stream_;
  std::streambuf* buf_;
  cb_func_ptr cb_func_;
  void* data_;
};

class LogWidget : public QWidget {
 public:
  explicit LogWidget(QWidget* parent, int max_num_blocks = 100000);
  ~LogWidget();

  void Append(const std::string& text);
  void Flush();
  void Clear();

 private:
  static void Update(const char* text,
                     std::streamsize count,
                     void* text_box_ptr);

  void SaveLog();

  QMutex mutex_;
  std::string text_queue_;
  QPlainTextEdit* text_box_;
  std::ofstream log_file_;
  StandardOutputRedirector<char, std::char_traits<char>>* cout_redirector_;
  StandardOutputRedirector<char, std::char_traits<char>>* cerr_redirector_;
  StandardOutputRedirector<char, std::char_traits<char>>* clog_redirector_;
};

}  // namespace colmap
