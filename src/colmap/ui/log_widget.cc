// SPDX-License-Identifier: BSD-3-Clause

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
    std::copy(message, message + message_len, text.begin() + severity_len);
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
      default:
        break;
    }
    log_widget_->Append(severity, std::move(text));
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

void LogWidget::Append(google::LogSeverity severity, std::string text) {
  QMutexLocker locker(&mutex_);
  text_queue_.push_back({severity, std::move(text)});
}

void LogWidget::Flush() {
  QMutexLocker locker(&mutex_);

  if (text_queue_.empty()) {
    return;
  }

  text_box_->moveCursor(QTextCursor::End);

#if COLMAP_GLOG_HAS_COLOR_SUPPORT
  if (FLAGS_colorlogtostderr) {
    for (const auto& entry : text_queue_) {
      QTextCharFormat format;
      switch (entry.severity) {
        case google::GLOG_WARNING:
          format.setForeground(Qt::darkYellow);
          format.setFontWeight(QFont::DemiBold);
          break;
        case google::GLOG_ERROR:
          format.setForeground(Qt::darkRed);
          format.setFontWeight(QFont::DemiBold);
          break;
        default:
          break;
      }
      text_box_->setCurrentCharFormat(format);
      text_box_->insertPlainText(QString::fromStdString(entry.text));
    }
  } else
#endif
  {
    for (const auto& entry : text_queue_) {
      text_box_->insertPlainText(QString::fromStdString(entry.text));
    }
  }

  text_box_->moveCursor(QTextCursor::End);
  text_queue_.clear();
}

void LogWidget::Clear() {
  QMutexLocker locker(&mutex_);
  text_queue_.clear();
  text_box_->clear();
}

}  // namespace colmap
