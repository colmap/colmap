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

#include "ui/log_widget.h"

namespace colmap {

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

  // Comment these lines if debugging, otherwise debug messages won't appear
  // on the console and the output is lost in the log widget when crashing
  cout_redirector_ = new StandardOutputRedirector<char, std::char_traits<char>>(
      std::cout, LogWidget::Update, this);
  cerr_redirector_ = new StandardOutputRedirector<char, std::char_traits<char>>(
      std::cerr, LogWidget::Update, this);
  clog_redirector_ = new StandardOutputRedirector<char, std::char_traits<char>>(
      std::clog, LogWidget::Update, this);

  QHBoxLayout* left_button_layout = new QHBoxLayout();

  QPushButton* save_log_button = new QPushButton(tr("Save"), this);
  connect(save_log_button, &QPushButton::released, this, &LogWidget::SaveLog);
  left_button_layout->addWidget(save_log_button);

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
  text_box_->setFont(QFont("Courier", 10));
  grid->addWidget(text_box_, 1, 0, 1, 2);
}

LogWidget::~LogWidget() {
  if (log_file_.is_open()) {
    log_file_.close();
  }

  delete cout_redirector_;
  delete cerr_redirector_;
  delete clog_redirector_;
}

void LogWidget::Append(const std::string& text) {
  QMutexLocker locker(&mutex_);
  text_queue_ += text;

  // Dump to log file
  if (log_file_.is_open()) {
    log_file_ << text;
  }
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

void LogWidget::Update(const char* text, std::streamsize count,
                       void* log_widget_ptr) {
  std::string text_str;
  for (std::streamsize i = 0; i < count; ++i) {
    if (text[i] == '\n') {
      text_str += "\n";
    } else {
      text_str += text[i];
    }
  }

  LogWidget* log_widget = static_cast<LogWidget*>(log_widget_ptr);
  log_widget->Append(text_str);
}

void LogWidget::SaveLog() {
  const std::string log_path =
      QFileDialog::getSaveFileName(this, tr("Select path to log file"), "",
                                   tr("Log (*.log)"))
          .toUtf8()
          .constData();

  if (log_path == "") {
    return;
  }

  std::ofstream file(log_path, std::ios::app);
  CHECK(file.is_open()) << log_path;
  file << text_box_->toPlainText().toUtf8().constData();
}

}  // namespace colmap
