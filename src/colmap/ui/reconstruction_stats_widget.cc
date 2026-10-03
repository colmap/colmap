// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/reconstruction_stats_widget.h"

namespace colmap {

ReconstructionStatsWidget::ReconstructionStatsWidget(QWidget* parent)
    : QWidget(parent) {
  setWindowFlags(Qt::Window);
  resize(parent->width() - 20, parent->height() - 20);
  setWindowTitle("Reconstruction statistics");

  stats_table_ = new QTableWidget(this);
  stats_table_->setColumnCount(2);
  stats_table_->horizontalHeader()->setVisible(false);
  stats_table_->verticalHeader()->setVisible(false);
  stats_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

  QGridLayout* grid = new QGridLayout(this);
  grid->addWidget(stats_table_);
}

void ReconstructionStatsWidget::Show(const Reconstruction& reconstruction) {
  QString stats;

  stats_table_->clearContents();
  stats_table_->setRowCount(0);

  AddStatistic("Cameras", QString::number(reconstruction.NumCameras()));
  AddStatistic("Frames", QString::number(reconstruction.NumFrames()));
  AddStatistic("Registered frames",
               QString::number(reconstruction.NumRegFrames()));
  AddStatistic("Images", QString::number(reconstruction.NumImages()));
  AddStatistic("Points", QString::number(reconstruction.NumPoints3D()));
  AddStatistic("Observations",
               QString::number(reconstruction.ComputeNumObservations()));
  AddStatistic("Mean track length",
               QString::number(reconstruction.ComputeMeanTrackLength()));
  AddStatistic(
      "Mean observations per image",
      QString::number(reconstruction.ComputeMeanObservationsPerRegImage()));
  AddStatistic("Mean reprojection error",
               QString::number(reconstruction.ComputeMeanReprojectionError()));
}

void ReconstructionStatsWidget::AddStatistic(const QString& header,
                                             const QString& content) {
  const int row = stats_table_->rowCount();
  stats_table_->insertRow(row);
  stats_table_->setItem(row, 0, new QTableWidgetItem(header));
  stats_table_->setItem(row, 1, new QTableWidgetItem(content));
}

}  // namespace colmap
