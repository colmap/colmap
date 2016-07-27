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

#include "ui/model_stats_widget.h"

namespace colmap {

ModelStatsWidget::ModelStatsWidget(QWidget* parent) : QWidget(parent) {
  setWindowFlags(Qt::Window);
  resize(parent->width() - 20, parent->height() - 20);
  setWindowTitle("Model statistics");

  stats_table_ = new QTableWidget(this);
  stats_table_->setColumnCount(2);
  stats_table_->horizontalHeader()->setVisible(false);
  stats_table_->verticalHeader()->setVisible(false);
  stats_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

  QGridLayout* grid = new QGridLayout(this);
  grid->addWidget(stats_table_);
}

void ModelStatsWidget::Update(const Reconstruction& model) {
  QString stats;

  stats_table_->clearContents();
  stats_table_->setRowCount(0);

  AddStatistic("Cameras", QString::number(model.NumCameras()));
  AddStatistic("Images", QString::number(model.NumImages()));
  AddStatistic("Registered images", QString::number(model.NumRegImages()));
  AddStatistic("Points", QString::number(model.NumPoints3D()));
  AddStatistic("Observations", QString::number(model.ComputeNumObservations()));
  AddStatistic("Mean track length",
               QString::number(model.ComputeMeanTrackLength()));
  AddStatistic("Mean observations per image",
               QString::number(model.ComputeMeanObservationsPerRegImage()));
  AddStatistic("Mean point error",
               QString::number(model.ComputeMeanReprojectionError()));
}

void ModelStatsWidget::AddStatistic(const QString& header,
                                    const QString& content) {
  const int row = stats_table_->rowCount();
  stats_table_->insertRow(row);
  stats_table_->setItem(row, 0, new QTableWidgetItem(header));
  stats_table_->setItem(row, 1, new QTableWidgetItem(content));
}

}  // namespace colmap
