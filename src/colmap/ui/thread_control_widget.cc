// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/bundle_adjustment_widget.h"

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
                                      const bool stoppable,
                                      std::unique_ptr<Thread> thread) {
  THROW_CHECK(!thread_);
  THROW_CHECK_NOTNULL(thread);

  thread_ = std::move(thread);

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
    connect(progress_bar_, &QProgressDialog::canceled, [this]() {
      destructor_->trigger();
    });
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

  StartThread(progress_text, false, std::make_unique<FunctionThread>(func));
}

}  // namespace colmap
