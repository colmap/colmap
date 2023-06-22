// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
  AddStatistic("Images", QString::number(reconstruction.NumImages()));
  AddStatistic("Registered images",
               QString::number(reconstruction.NumRegImages()));
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
