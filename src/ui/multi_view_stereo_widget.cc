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

#include "ui/multi_view_stereo_widget.h"

#include <boost/filesystem.hpp>

#include "base/undistortion.h"
#include "mvs/patch_match.h"
#include "util/misc.h"

namespace colmap {

MultiViewStereoOptionsWidget::MultiViewStereoOptionsWidget(
    QWidget* parent, OptionManager* options)
    : OptionsWidget(parent) {
  setWindowTitle("Multi-view stereo options");

  AddOptionInt(&options->dense_mapper_options->max_image_size, "max_image_size",
               0);
  AddOptionInt(&options->dense_mapper_options->patch_match.gpu_index,
               "gpu_index", -1);
  AddOptionInt(&options->dense_mapper_options->patch_match.window_radius,
               "window_radius");
  AddOptionDouble(&options->dense_mapper_options->patch_match.sigma_spatial,
                  "sigma_spatial");
  AddOptionDouble(&options->dense_mapper_options->patch_match.sigma_color,
                  "sigma_color");
  AddOptionInt(&options->dense_mapper_options->patch_match.num_samples,
               "num_samples");
  AddOptionDouble(&options->dense_mapper_options->patch_match.ncc_sigma,
                  "ncc_sigma");
  AddOptionDouble(
      &options->dense_mapper_options->patch_match.min_triangulation_angle,
      "min_triangulation_angle");
  AddOptionDouble(
      &options->dense_mapper_options->patch_match.incident_angle_sigma,
      "incident_angle_sigma");
  AddOptionInt(&options->dense_mapper_options->patch_match.num_iterations,
               "num_iterations");
  AddOptionDouble(
      &options->dense_mapper_options->patch_match.geom_consistency_regularizer,
      "geom_consistency_regularizer");
  AddOptionDouble(
      &options->dense_mapper_options->patch_match.geom_consistency_max_cost,
      "geom_consistency_max_cost");
  AddOptionDouble(&options->dense_mapper_options->patch_match.filter_min_ncc,
                  "filter_min_ncc");
  AddOptionDouble(&options->dense_mapper_options->patch_match
                       .filter_min_triangulation_angle,
                  "filter_min_triangulation_angle");
  AddOptionInt(
      &options->dense_mapper_options->patch_match.filter_min_num_consistent,
      "filter_min_num_consistent");
  AddOptionDouble(&options->dense_mapper_options->patch_match
                       .filter_min_triangulation_angle,
                  "filter_min_triangulation_angle");
  AddOptionDouble(&options->dense_mapper_options->patch_match
                       .filter_geom_consistency_max_cost,
                  "filter_geom_consistency_max_cost");
}

MultiViewStereoWidget::MultiViewStereoWidget(QWidget* parent,
                                             OptionManager* options)
    : QWidget(parent),
      options_(options),
      thread_control_widget_(new ThreadControlWidget(this)),
      options_widget_(new MultiViewStereoOptionsWidget(this, options)) {
  setWindowFlags(Qt::Window);
  setWindowTitle("Multi-view stereo");
  resize(parent->size().width() - 340, parent->size().height() - 20);

  QGridLayout* grid = new QGridLayout(this);

  QPushButton* prepare_button = new QPushButton(tr("Prepare"), this);
  connect(prepare_button, &QPushButton::released, this,
          &MultiViewStereoWidget::Prepare);
  grid->addWidget(prepare_button, 0, 0, Qt::AlignLeft);

  run_button_ = new QPushButton(tr("Run"), this);
  connect(run_button_, &QPushButton::released, this,
          &MultiViewStereoWidget::Run);
  grid->addWidget(run_button_, 0, 1, Qt::AlignLeft);

  QPushButton* options_button = new QPushButton(tr("Options"), this);
  connect(options_button, &QPushButton::released, options_widget_,
          &OptionsWidget::show);
  grid->addWidget(options_button, 0, 2, Qt::AlignLeft);

  QLabel* workspace_path_label = new QLabel("Workspace", this);
  grid->addWidget(workspace_path_label, 0, 3, Qt::AlignRight);

  workspace_path_text_ = new QLineEdit(this);
  grid->addWidget(workspace_path_text_, 0, 4, Qt::AlignRight);

  QPushButton* workspace_path_button = new QPushButton(tr("Select"), this);
  connect(workspace_path_button, &QPushButton::released, this,
          &MultiViewStereoWidget::SelectWorkspacePath);
  grid->addWidget(workspace_path_button, 0, 5, Qt::AlignRight);

  QStringList table_header;
  table_header << "image_id"
               << "image_name"
               << "photometric"
               << "geometric";

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(table_header.size());
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setDefaultSectionSize(20);

  grid->addWidget(table_widget_, 1, 0, 1, 6);

  grid->setColumnStretch(2, 1);

  RefreshWorkspace();
}

void MultiViewStereoWidget::Show(Reconstruction* reconstruction) {
  reconstruction_ = reconstruction;
  show();
  raise();
}

void MultiViewStereoWidget::Prepare() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  COLMAPUndistorter* undistorter =
      new COLMAPUndistorter(UndistortCameraOptions(), *reconstruction_,
                            *options_->image_path, workspace_path);
  thread_control_widget_->StartThread("Preparing...", true, undistorter);
}

void MultiViewStereoWidget::Run() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

#ifdef CUDA_ENABLED
  mvs::PatchMatchProcessor* processor = new mvs::PatchMatchProcessor(
      options_->dense_mapper_options->patch_match, workspace_path, "COLMAP",
      options_->dense_mapper_options->max_image_size);
  thread_control_widget_->StartThread("Processing...", true, processor);
#else
  QMessageBox::critical(this, "", tr("CUDA not supported"));
#endif
}

void MultiViewStereoWidget::SelectWorkspacePath() {
  std::string directory_path = "";
  if (workspace_path_text_->text().isEmpty()) {
    directory_path =
        boost::filesystem::path(*options_->project_path).parent_path().string();
  }

  workspace_path_text_->setText(QFileDialog::getExistingDirectory(
      this, tr("Select workspace path..."),
      QString::fromStdString(directory_path), QFileDialog::ShowDirsOnly));

  RefreshWorkspace();
}

std::string MultiViewStereoWidget::GetWorkspacePath() {
  const std::string workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (boost::filesystem::is_directory(workspace_path)) {
    return workspace_path;
  } else {
    QMessageBox::critical(this, "", tr("Invalid workspace path"));
    return "";
  }
}

void MultiViewStereoWidget::RefreshWorkspace() {
  const std::string workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (!boost::filesystem::is_directory(workspace_path)) {
    run_button_->setEnabled(false);
    return;
  }

  if (boost::filesystem::is_directory(JoinPaths(workspace_path, "images")) &&
      boost::filesystem::is_directory(JoinPaths(workspace_path, "sparse")) &&
      boost::filesystem::is_directory(
          JoinPaths(workspace_path, "dense/depth_maps")) &&
      boost::filesystem::is_directory(
          JoinPaths(workspace_path, "dense/normal_maps")) &&
      boost::filesystem::is_directory(
          JoinPaths(workspace_path, "dense/consistency_graphs")) &&
      boost::filesystem::exists(
          JoinPaths(workspace_path, "dense/patch-match.cfg"))) {
    run_button_->setEnabled(true);
  } else {
    run_button_->setEnabled(false);
    return;
  }
}

}  // namespace colmap
