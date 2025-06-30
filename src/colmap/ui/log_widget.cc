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

#include "colmap/ui/log_widget.h"

namespace colmap {
namespace {

class GlogSink : public google::LogSink {
 public:
  explicit GlogSink(LogWidget* log_widget)
      : log_widget_(THROW_CHECK_NOTNULL(log_widget)) {
    google::AddLogSink(this);
  }

  ~GlogSink() { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const struct ::tm* /*tm_time*/,
            const char* message,
            size_t message_len) override {
    if (severity != google::GLOG_INFO && severity != google::GLOG_WARNING &&
        severity != google::GLOG_ERROR) {
      return;
    }
    const int severity_len =
        (severity == google::GLOG_WARNING || severity == google::GLOG_ERROR)
            ? 3
            : 0;
    std::string text(message_len + 1 + severity_len, '\0');
    std::copy(message + severity_len,
              message + message_len + severity_len,
              text.begin());
    text.back() = '\n';
    switch (severity) {
      case google::GLOG_WARNING:
        text[0] = 'W';
        text[1] = ':';
        text[2] = ' ';
        break;
      case google::GLOG_ERROR:
        text[0] = 'E';
        text[1] = ':';
        text[2] = ' ';
        break;
    }
    log_widget_->Append(text);
  }

 private:
  LogWidget* log_widget_;
};

}  // namespace

LogWidget::LogWidget(QWidget* parent, const int max_num_blocks) {
  setWindowFlags(Qt::Window);
  setWindowTitle("Log");
  resize(320, parent->height());

  QGridLayout* grid = new QGridLayout(this);
  grid->setContentsMargins(5, 10, 5, 5);

  qRegisterMetaType<QTextCursor>("QTextCursor");
  qRegisterMetaType<QTextBlock>("QTextBlock");

  QTimer* timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &LogWidget::Flush);
  timer->start(100);

  log_sink_ = std::make_unique<GlogSink>(this);

  QHBoxLayout* left_button_layout = new QHBoxLayout();

  QPushButton* clear_button = new QPushButton(tr("Clear"), this);
  connect(clear_button, &QPushButton::released, this, &LogWidget::Clear);
  left_button_layout->addWidget(clear_button);

  grid->addLayout(left_button_layout, 0, 0, Qt::AlignLeft);

  QHBoxLayout* right_button_layout = new QHBoxLayout();

  grid->addLayout(right_button_layout, 0, 1, Qt::AlignRight);

  text_box_ = new QPlainTextEdit(this);
  text_box_->setReadOnly(true);
  text_box_->setMaximumBlockCount(max_num_blocks);
  text_box_->setWordWrapMode(QTextOption::NoWrap);
  QFont font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  font.setPointSize(10);
  text_box_->setFont(font);
  grid->addWidget(text_box_, 1, 0, 1, 2);
}

void LogWidget::Append(const std::string& text) {
  QMutexLocker locker(&mutex_);
  text_queue_ += text;
}

void LogWidget::Flush() {
  QMutexLocker locker(&mutex_);

  if (text_queue_.size() > 0) {
    // Write to log widget
    text_box_->moveCursor(QTextCursor::End);
    text_box_->insertPlainText(QString::fromStdString(text_queue_));
    text_box_->moveCursor(QTextCursor::End);
    text_queue_.clear();
  }
}

void LogWidget::Clear() {
  QMutexLocker locker(&mutex_);
  text_queue_.clear();
  text_box_->clear();
}

}  // namespace colmap
