// COLMAP - Structure-from-Motion and Multi-View Stereo.
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
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "ui/main_window.h"
#include "util/misc.h"

namespace colmap {
namespace {

const static std::string kFusedFileName = "fused.ply";
const static std::string kMeshedFileName = "meshed.ply";

class PatchMatchOptionsTab : public OptionsWidget {
 public:
  PatchMatchOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    // Set a relatively small default image size to avoid too long computation.
    if (options->dense_mapper_options->max_image_size == 0) {
      options->dense_mapper_options->max_image_size = 2000;
    }

    AddOptionInt(&options->dense_mapper_options->max_image_size,
                 "max_image_size", 0);
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
    AddOptionDouble(&options->dense_mapper_options->patch_match
                         .geom_consistency_regularizer,
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
};

class StereoFusionOptionsTab : public OptionsWidget {
 public:
  StereoFusionOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(&options->dense_mapper_options->fusion.min_num_pixels,
                 "min_num_pixels", 0);
    AddOptionInt(&options->dense_mapper_options->fusion.max_num_pixels,
                 "max_num_pixels", 0);
    AddOptionInt(&options->dense_mapper_options->fusion.max_traversal_depth,
                 "max_traversal_depth", 1);
    AddOptionDouble(&options->dense_mapper_options->fusion.max_reproj_error,
                    "max_reproj_error", 0);
    AddOptionDouble(&options->dense_mapper_options->fusion.max_depth_error,
                    "max_depth_error", 0, 1, 0.0001, 4);
    AddOptionDouble(&options->dense_mapper_options->fusion.max_normal_error,
                    "max_normal_error", 0, 180);
  }
};

class PoissonReconstructionOptionsTab : public OptionsWidget {
 public:
  PoissonReconstructionOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionDouble(&options->dense_mapper_options->poisson.point_weight,
                    "point_weight", 0);
    AddOptionInt(&options->dense_mapper_options->poisson.depth, "depth", 1);
    AddOptionDouble(&options->dense_mapper_options->poisson.trim, "trim", 0);
  }
};

// Read the specified reference image names from a patch match configuration.
std::vector<std::string> ReadRefImageNamesFromConfig(
    const std::string& config_path) {
  std::ifstream file(config_path);
  CHECK(file.is_open());

  std::string line;
  std::string ref_image_name;
  std::vector<std::string> ref_image_names;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = line;
      ref_image_names.push_back(ref_image_name);
      continue;
    } else {
      ref_image_name.clear();
    }
  }

  return ref_image_names;
}

}  // namepspace

MultiViewStereoOptionsWidget::MultiViewStereoOptionsWidget(
    QWidget* parent, OptionManager* options)
    : QWidget(parent) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Multi-view stereo options");

  QGridLayout* grid = new QGridLayout(this);

  QTabWidget* tab_widget = new QTabWidget(this);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget->addTab(new PatchMatchOptionsTab(this, options), "Stereo");
  tab_widget->addTab(new StereoFusionOptionsTab(this, options), "Fusion");
  tab_widget->addTab(new PoissonReconstructionOptionsTab(this, options),
                     "Meshing");

  grid->addWidget(tab_widget, 0, 0);
}

MultiViewStereoWidget::MultiViewStereoWidget(MainWindow* main_window,
                                             OptionManager* options)
    : QWidget(main_window),
      main_window_(main_window),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)),
      options_widget_(new MultiViewStereoOptionsWidget(this, options)),
      photometric_done_(false),
      geometric_done_(false) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Multi-view stereo");
  resize(main_window_->size().width() - 200,
         main_window_->size().height() - 20);

  QGridLayout* grid = new QGridLayout(this);

  undistortion_button_ = new QPushButton(tr("Undistortion"), this);
  connect(undistortion_button_, &QPushButton::released, this,
          &MultiViewStereoWidget::Undistort);
  grid->addWidget(undistortion_button_, 0, 0, Qt::AlignLeft);

  stereo_button_ = new QPushButton(tr("Stereo"), this);
  connect(stereo_button_, &QPushButton::released, this,
          &MultiViewStereoWidget::Stereo);
  grid->addWidget(stereo_button_, 0, 1, Qt::AlignLeft);

  fusion_button_ = new QPushButton(tr("Fusion"), this);
  connect(fusion_button_, &QPushButton::released, this,
          &MultiViewStereoWidget::Fusion);
  grid->addWidget(fusion_button_, 0, 2, Qt::AlignLeft);

  meshing_button_ = new QPushButton(tr("Meshing"), this);
  connect(meshing_button_, &QPushButton::released, this,
          &MultiViewStereoWidget::Meshing);
  grid->addWidget(meshing_button_, 0, 3, Qt::AlignLeft);

  QPushButton* options_button = new QPushButton(tr("Options"), this);
  connect(options_button, &QPushButton::released, options_widget_,
          &OptionsWidget::show);
  grid->addWidget(options_button, 0, 4, Qt::AlignLeft);

  QLabel* workspace_path_label = new QLabel("Workspace", this);
  grid->addWidget(workspace_path_label, 0, 5, Qt::AlignRight);

  workspace_path_text_ = new QLineEdit(this);
  grid->addWidget(workspace_path_text_, 0, 6, Qt::AlignRight);
  connect(workspace_path_text_, &QLineEdit::textChanged, this,
          &MultiViewStereoWidget::RefreshWorkspace);

  QPushButton* workspace_path_button = new QPushButton(tr("Select"), this);
  connect(workspace_path_button, &QPushButton::released, this,
          &MultiViewStereoWidget::SelectWorkspacePath);
  grid->addWidget(workspace_path_button, 0, 7, Qt::AlignRight);

  QStringList table_header;
  table_header << "image_name"
               << ""
               << "photometric"
               << "geometric";

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(table_header.size());
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->verticalHeader()->setDefaultSectionSize(25);

  grid->addWidget(table_widget_, 1, 0, 1, 8);

  grid->setColumnStretch(4, 1);

  image_viewer_widget_ = new ImageViewerWidget(this);
  image_viewer_widget_->setWindowFlags(Qt::Dialog);
  image_viewer_widget_->setWindowModality(Qt::ApplicationModal);

  refresh_workspace_action_ = new QAction(this);
  connect(refresh_workspace_action_, &QAction::triggered, this,
          &MultiViewStereoWidget::RefreshWorkspace);

  write_fused_points_action_ = new QAction(this);
  connect(write_fused_points_action_, &QAction::triggered, this,
          &MultiViewStereoWidget::WriteFusedPoints);

  RefreshWorkspace();
}

void MultiViewStereoWidget::Show(Reconstruction* reconstruction) {
  reconstruction_ = reconstruction;
  show();
  raise();
}

void MultiViewStereoWidget::Undistort() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  if (reconstruction_ == nullptr || reconstruction_->NumRegImages() < 2) {
    QMessageBox::critical(this, "",
                          tr("No reconstruction selected in main window"));
  }

  COLMAPUndistorter* undistorter =
      new COLMAPUndistorter(UndistortCameraOptions(), *reconstruction_,
                            *options_->image_path, workspace_path);
  undistorter->AddCallback(Thread::FINISHED_CALLBACK,
                           [this]() { refresh_workspace_action_->trigger(); });
  thread_control_widget_->StartThread("Preparing...", true, undistorter);
}

void MultiViewStereoWidget::Stereo() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

#ifdef CUDA_ENABLED
  mvs::PatchMatchController* processor = new mvs::PatchMatchController(
      options_->dense_mapper_options->patch_match, workspace_path, "COLMAP",
      options_->dense_mapper_options->max_image_size);
  processor->AddCallback(Thread::FINISHED_CALLBACK,
                         [this]() { refresh_workspace_action_->trigger(); });
  thread_control_widget_->StartThread("Processing...", true, processor);
#else
  QMessageBox::critical(this, "", tr("CUDA not supported"));
#endif
}

void MultiViewStereoWidget::Fusion() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  std::string input_type;
  if (geometric_done_) {
    input_type = "geometric";
  } else if (photometric_done_) {
    input_type = "photometric";
  } else {
    QMessageBox::critical(this, "",
                          tr("All images must be processed prior to fusion"));
  }

  mvs::StereoFusion* fuser =
      new mvs::StereoFusion(options_->dense_mapper_options->fusion,
                            workspace_path, "COLMAP", input_type);
  fuser->AddCallback(Thread::FINISHED_CALLBACK, [this, fuser]() {
    fused_points_ = fuser->GetFusedPoints();
    write_fused_points_action_->trigger();
  });
  thread_control_widget_->StartThread("Fusing...", true, fuser);
}

void MultiViewStereoWidget::Meshing() {
  const std::string workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  if (boost::filesystem::exists(JoinPaths(workspace_path, kFusedFileName))) {
    thread_control_widget_->StartFunction(
        "Meshing...", [this, workspace_path]() {
          mvs::PoissonReconstruction(options_->dense_mapper_options->poisson,
                                     JoinPaths(workspace_path, kFusedFileName),
                                     kMeshedFileName);
        });
  }
}

void MultiViewStereoWidget::SelectWorkspacePath() {
  std::string workspace_path;
  if (workspace_path_text_->text().isEmpty()) {
    workspace_path =
        boost::filesystem::path(*options_->project_path).parent_path().string();
  } else {
    workspace_path = workspace_path_text_->text().toUtf8().constData();
  }

  workspace_path_text_->setText(QFileDialog::getExistingDirectory(
      this, tr("Select workspace path..."),
      QString::fromStdString(workspace_path), QFileDialog::ShowDirsOnly));

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
  table_widget_->clearContents();
  table_widget_->setRowCount(0);

  const std::string workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (boost::filesystem::is_directory(workspace_path)) {
    undistortion_button_->setEnabled(true);
  } else {
    undistortion_button_->setEnabled(false);
    stereo_button_->setEnabled(false);
    fusion_button_->setEnabled(false);
    meshing_button_->setEnabled(false);
    return;
  }

  images_path_ = JoinPaths(workspace_path, "images");
  depth_maps_path_ = JoinPaths(workspace_path, "dense/depth_maps");
  normal_maps_path_ = JoinPaths(workspace_path, "dense/normal_maps");
  const std::string config_path =
      JoinPaths(workspace_path, "dense/patch-match.cfg");

  if (boost::filesystem::is_directory(images_path_) &&
      boost::filesystem::is_directory(depth_maps_path_) &&
      boost::filesystem::is_directory(normal_maps_path_) &&
      boost::filesystem::is_directory(JoinPaths(workspace_path, "sparse")) &&
      boost::filesystem::is_directory(
          JoinPaths(workspace_path, "dense/consistency_graphs")) &&
      boost::filesystem::exists(config_path)) {
    stereo_button_->setEnabled(true);
  } else {
    stereo_button_->setEnabled(false);
    fusion_button_->setEnabled(false);
    meshing_button_->setEnabled(false);
    return;
  }

  const std::vector<std::string> image_names =
      ReadRefImageNamesFromConfig(config_path);
  table_widget_->setRowCount(image_names.size());

  for (size_t i = 0; i < image_names.size(); ++i) {
    const std::string image_name = image_names[i];
    const std::string image_path = JoinPaths(images_path_, image_name);

    QTableWidgetItem* image_name_item =
        new QTableWidgetItem(QString::fromStdString(image_name));
    table_widget_->setItem(i, 0, image_name_item);

    QPushButton* image_button = new QPushButton("Image");
    connect(image_button, &QPushButton::released,
            [this, image_name, image_path]() {
              image_viewer_widget_->setWindowTitle(
                  QString("Image for %1").arg(image_name.c_str()));
              image_viewer_widget_->ReadAndShow(image_path, true);
            });
    table_widget_->setCellWidget(i, 1, image_button);

    table_widget_->setCellWidget(
        i, 2, GenerateTableButtonWidget(image_name, "photometric"));
    table_widget_->setCellWidget(
        i, 3, GenerateTableButtonWidget(image_name, "geometric"));
  }

  table_widget_->resizeColumnsToContents();

  fusion_button_->setEnabled(photometric_done_ || geometric_done_);
  meshing_button_->setEnabled(
      boost::filesystem::exists(JoinPaths(workspace_path, kFusedFileName)));
}

void MultiViewStereoWidget::WriteFusedPoints() {
  const int reply = QMessageBox::question(
      this, "", tr("Do you want to visualize the point cloud?"),
      QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    main_window_->ImportFusedPoints(fused_points_);
  }

  const std::string workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (workspace_path.empty()) {
    fused_points_ = {};
    return;
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, workspace_path]() {
        mvs::WritePlyBinary(JoinPaths(workspace_path, kFusedFileName),
                            fused_points_);
        fused_points_ = {};
      });
}

QWidget* MultiViewStereoWidget::GenerateTableButtonWidget(
    const std::string& image_name, const std::string& type) {
  CHECK(type == "photometric" || type == "geometric");
  const bool photometric = type == "photometric";

  if (photometric) {
    photometric_done_ = true;
  } else {
    geometric_done_ = true;
  }

  const std::string depth_map_path =
      JoinPaths(depth_maps_path_,
                StringPrintf("%s.%s.bin", image_name.c_str(), type.c_str()));
  const std::string normal_map_path =
      JoinPaths(normal_maps_path_,
                StringPrintf("%s.%s.bin", image_name.c_str(), type.c_str()));

  QWidget* button_widget = new QWidget();
  QGridLayout* button_layout = new QGridLayout(button_widget);
  button_layout->setContentsMargins(0, 0, 0, 0);

  QPushButton* depth_map_button = new QPushButton("Depth map", button_widget);
  if (boost::filesystem::exists(depth_map_path)) {
    connect(depth_map_button, &QPushButton::released,
            [this, image_name, depth_map_path]() {
              mvs::DepthMap depth_map;
              depth_map.Read(depth_map_path);
              image_viewer_widget_->setWindowTitle(
                  QString("Depth map for %1").arg(image_name.c_str()));
              image_viewer_widget_->ShowBitmap(depth_map.ToBitmap(2, 98), true);
            });
  } else {
    depth_map_button->setEnabled(false);
    if (photometric) {
      photometric_done_ = false;
    } else {
      geometric_done_ = false;
    }
  }
  button_layout->addWidget(depth_map_button, 0, 1, Qt::AlignLeft);

  QPushButton* normal_map_button = new QPushButton("Normal map", button_widget);
  if (boost::filesystem::exists(normal_map_path)) {
    connect(normal_map_button, &QPushButton::released,
            [this, image_name, normal_map_path]() {
              mvs::NormalMap normal_map;
              normal_map.Read(normal_map_path);
              image_viewer_widget_->setWindowTitle(
                  QString("Normal map for %1").arg(image_name.c_str()));
              image_viewer_widget_->ShowBitmap(normal_map.ToBitmap(), true);
            });
  } else {
    normal_map_button->setEnabled(false);
    if (photometric) {
      photometric_done_ = false;
    } else {
      geometric_done_ = false;
    }
  }
  button_layout->addWidget(normal_map_button, 0, 2, Qt::AlignLeft);

  return button_widget;
}

}  // namespace colmap
