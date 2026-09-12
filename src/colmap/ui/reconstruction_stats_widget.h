// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"

#include <QtWidgets>

namespace colmap {

class ReconstructionStatsWidget : public QWidget {
 public:
  explicit ReconstructionStatsWidget(QWidget* parent);

  void Show(const Reconstruction& reconstruction);

 private:
  void AddStatistic(const QString& header, const QString& content);

  QTableWidget* stats_table_;
};

}  // namespace colmap
