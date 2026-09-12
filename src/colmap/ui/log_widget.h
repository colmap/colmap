// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"

#include <QtGui>
#include <QtWidgets>

namespace colmap {

class LogWidget : public QWidget {
 public:
  explicit LogWidget(QWidget* parent, int max_num_blocks = 100000);

  void Append(google::LogSeverity severity, std::string text);
  void Flush();
  void Clear();

 private:
  struct LogEntry {
    google::LogSeverity severity;
    std::string text;
  };

  QMutex mutex_;
  std::vector<LogEntry> text_queue_;
  QPlainTextEdit* text_box_;
  std::unique_ptr<google::LogSink> log_sink_;
};

}  // namespace colmap
