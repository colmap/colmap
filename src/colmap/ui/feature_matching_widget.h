// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class FeatureMatchingWidget : public QWidget {
 public:
  FeatureMatchingWidget(QWidget* parent, OptionManager* options);

 private:
  void showEvent(QShowEvent* event);
  void hideEvent(QHideEvent* event);

  QWidget* parent_;
  QTabWidget* tab_widget_;
};

}  // namespace colmap
