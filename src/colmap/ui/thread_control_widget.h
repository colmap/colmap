// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/threading.h"

#include <QtCore>
#include <QtWidgets>
#include <memory>

namespace colmap {

class ThreadControlWidget : public QWidget {
 public:
  explicit ThreadControlWidget(QWidget* parent);

  void StartThread(const QString& progress_text,
                   bool stoppable,
                   std::unique_ptr<Thread> thread);
  void StartFunction(const QString& progress_text,
                     const std::function<void()>& func);

 private:
  QProgressDialog* progress_bar_;
  QAction* destructor_;
  std::unique_ptr<Thread> thread_;
};

}  // namespace colmap
