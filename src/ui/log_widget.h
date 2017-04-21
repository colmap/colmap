// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_UI_LOG_WIDGET_H_
#define COLMAP_SRC_UI_LOG_WIDGET_H_

#include <iostream>

#include <QtGui>
#include <QtWidgets>

#include "util/option_manager.h"

namespace colmap {

template <class Elem = char, class Tr = std::char_traits<Elem>>
class StandardOutputRedirector : public std::basic_streambuf<Elem, Tr> {
  typedef void (*cb_func_ptr)(const Elem*, std::streamsize count, void* data);

 public:
  StandardOutputRedirector(std::ostream& stream, cb_func_ptr cb_func,
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
  LogWidget(QWidget* parent, const int max_num_blocks = 100000);
  ~LogWidget();

  void Append(const std::string& text);
  void Flush();
  void Clear();

 private:
  static void Update(const char* text, std::streamsize count,
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

#endif  // COLMAP_SRC_UI_LOG_WIDGET_H_
