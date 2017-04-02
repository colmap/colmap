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

#include "ui/bundle_adjustment_widget.h"

namespace colmap {

ThreadControlWidget::ThreadControlWidget(QWidget* parent)
    : QWidget(parent),
      progress_bar_(nullptr),
      destructor_(new QAction(this)),
      thread_(nullptr) {
  connect(destructor_, &QAction::triggered, this, [this]() {
    if (thread_) {
      thread_->Stop();
      thread_->Wait();
      thread_.reset();
    }
    if (progress_bar_ != nullptr) {
      progress_bar_->hide();
    }
  });
}

void ThreadControlWidget::StartThread(const QString& progress_text,
                                      const bool stoppable, Thread* thread) {
  CHECK(!thread_);
  CHECK_NOTNULL(thread);

  thread_.reset(thread);

  if (progress_bar_ == nullptr) {
    progress_bar_ = new QProgressDialog(this);
    progress_bar_->setWindowModality(Qt::ApplicationModal);
    progress_bar_->setWindowFlags(Qt::Dialog | Qt::WindowTitleHint |
                                  Qt::CustomizeWindowHint);
    // Use a single space to clear the window title on Windows, otherwise it
    // will contain the name of the executable.
    progress_bar_->setWindowTitle(" ");
    progress_bar_->setLabel(new QLabel(this));
    progress_bar_->setMaximum(0);
    progress_bar_->setMinimum(0);
    progress_bar_->setValue(0);
    connect(progress_bar_, &QProgressDialog::canceled,
            [this]() { destructor_->trigger(); });
  }

  // Enable the cancel button if the thread is stoppable.
  QPushButton* cancel_button =
      progress_bar_->findChildren<QPushButton*>().at(0);
  cancel_button->setEnabled(stoppable);

  progress_bar_->setLabelText(progress_text);

  // Center the progress bar wrt. the parent widget.
  const QPoint global =
      parentWidget()->mapToGlobal(parentWidget()->rect().center());
  progress_bar_->move(global.x() - progress_bar_->width() / 2,
                      global.y() - progress_bar_->height() / 2);

  progress_bar_->show();
  progress_bar_->raise();

  thread_->AddCallback(Thread::FINISHED_CALLBACK,
                       [this]() { destructor_->trigger(); });
  thread_->Start();
}

void ThreadControlWidget::StartFunction(const QString& progress_text,
                                        const std::function<void()>& func) {
  class FunctionThread : public Thread {
   public:
    explicit FunctionThread(const std::function<void()>& f) : func_(f) {}

   private:
    void Run() { func_(); }
    const std::function<void()> func_;
  };

  StartThread(progress_text, false, new FunctionThread(func));
}

}  // namespace colmap
