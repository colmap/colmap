// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
