// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

void ThreadControlWidget::Start(const QString& progress_text,
                                Thread* thread) {
  CHECK(!thread_);
  CHECK_NOTNULL(thread);

  thread_.reset(thread);

  if (progress_bar_ == nullptr) {
    progress_bar_ = new QProgressDialog(this);
    progress_bar_->setWindowModality(Qt::ApplicationModal);
    progress_bar_->setMaximum(0);
    progress_bar_->setMinimum(0);
    progress_bar_->setValue(0);
    connect(progress_bar_, &QProgressDialog::canceled,
            [this]() { destructor_->trigger(); });
  }

  progress_bar_->setLabel(new QLabel(progress_text, this));
  progress_bar_->show();
  progress_bar_->raise();

  thread_->SetCallback(Thread::FINISHED_CALLBACK,
                       [this]() { destructor_->trigger(); });
  thread_->Start();
}

}  // namespace colmap
