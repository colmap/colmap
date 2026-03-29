// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/ui/dense_reconstruction_widget.h"

#include "colmap/controllers/undistorters.h"
#include "colmap/image/undistortion.h"
#if defined(COLMAP_MVS_ENABLED)
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#endif
#include "colmap/ui/main_window.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/controller_thread.h"

namespace colmap {
namespace {

const static std::string kFusedFileName = "fused.ply";
const static std::string kPoissonMeshedFileName = "meshed-poisson.ply";
const static std::string kDelaunayMeshedFileName = "meshed-delaunay.ply";

#if defined(COLMAP_MVS_ENABLED)
class StereoOptionsTab : public OptionsWidget {
 public:
  StereoOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    // Set a relatively small default image size to avoid too long computation.
    if (options->patch_match_stereo->max_image_size == -1) {
      options->patch_match_stereo->max_image_size = 2000;
    }

    AddOptionInt(
        &options->patch_match_stereo->max_image_size, "max_image_size", -1);
    AddOptionText(&options->patch_match_stereo->gpu_index, "gpu_index");
    AddOptionDouble(&options->patch_match_stereo->depth_min, "depth_min", -1);
    AddOptionDouble(&options->patch_match_stereo->depth_max, "depth_max", -1);
    AddOptionInt(&options->patch_match_stereo->window_radius, "window_radius");
    AddOptionInt(&options->patch_match_stereo->window_step, "window_step");
    AddOptionDouble(
        &options->patch_match_stereo->sigma_spatial, "sigma_spatial", -1);
    AddOptionDouble(&options->patch_match_stereo->sigma_color, "sigma_color");
    AddOptionInt(&options->patch_match_stereo->num_samples, "num_samples");
    AddOptionDouble(&options->patch_match_stereo->ncc_sigma, "ncc_sigma");
    AddOptionDouble(&options->patch_match_stereo->min_triangulation_angle,
                    "min_triangulation_angle");
    AddOptionDouble(&options->patch_match_stereo->incident_angle_sigma,
                    "incident_angle_sigma");
    AddOptionInt(&options->patch_match_stereo->num_iterations,
                 "num_iterations");
    AddOptionBool(&options->patch_match_stereo->geom_consistency,
                  "geom_consistency");
    AddOptionDouble(&options->patch_match_stereo->geom_consistency_regularizer,
                    "geom_consistency_regularizer");
    AddOptionDouble(&options->patch_match_stereo->geom_consistency_max_cost,
                    "geom_consistency_max_cost");
    AddOptionBool(&options->patch_match_stereo->filter, "filter");
    AddOptionDouble(&options->patch_match_stereo->filter_min_ncc,
                    "filter_min_ncc");
    AddOptionDouble(
        &options->patch_match_stereo->filter_min_triangulation_angle,
        "filter_min_triangulation_angle");
    AddOptionInt(&options->patch_match_stereo->filter_min_num_consistent,
                 "filter_min_num_consistent");
    AddOptionDouble(
        &options->patch_match_stereo->filter_geom_consistency_max_cost,
        "filter_geom_consistency_max_cost");
    AddOptionDouble(&options->patch_match_stereo->cache_size,
                    "cache_size [gigabytes]",
                    0,
                    std::numeric_limits<double>::max(),
                    0.1,
                    1);
    AddOptionBool(&options->patch_match_stereo->write_consistency_graph,
                  "write_consistency_graph");
    AddOptionInt(&options->patch_match_stereo->num_threads, "num_threads", -1);
  }
};

class FusionOptionsTab : public OptionsWidget {
 public:
  FusionOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(&options->stereo_fusion->max_image_size, "max_image_size", -1);
    AddOptionInt(&options->stereo_fusion->min_num_pixels, "min_num_pixels", 0);
    AddOptionInt(&options->stereo_fusion->max_num_pixels, "max_num_pixels", 0);
    AddOptionInt(
        &options->stereo_fusion->max_traversal_depth, "max_traversal_depth", 1);
    AddOptionDouble(
        &options->stereo_fusion->max_reproj_error, "max_reproj_error", 0);
    AddOptionDouble(&options->stereo_fusion->max_depth_error,
                    "max_depth_error",
                    0,
                    1,
                    0.0001,
                    4);
    AddOptionDouble(
        &options->stereo_fusion->max_normal_error, "max_normal_error", 0, 180);
    AddOptionInt(
        &options->stereo_fusion->check_num_images, "check_num_images", 1);
    AddOptionDouble(&options->stereo_fusion->cache_size,
                    "cache_size [gigabytes]",
                    0,
                    std::numeric_limits<double>::max(),
                    0.1,
                    1);
    AddOptionBool(&options->stereo_fusion->use_cache, "use_cache");
  }
};

class MeshingOptionsTab : public OptionsWidget {
 public:
  MeshingOptionsTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddSection("Poisson Meshing");
    AddOptionDouble(&options->poisson_meshing->point_weight, "point_weight", 0);
    AddOptionInt(&options->poisson_meshing->depth, "depth", 1);
    AddOptionBool(&options->poisson_meshing->color, "color");
    AddOptionDouble(&options->poisson_meshing->trim, "trim", 0);
    AddOptionInt(&options->poisson_meshing->num_threads, "num_threads", -1);

    AddSection("Delaunay Meshing");
    AddOptionDouble(
        &options->delaunay_meshing->max_proj_dist, "max_proj_dist", 0);
    AddOptionDouble(
        &options->delaunay_meshing->max_depth_dist, "max_depth_dist", 0);
    AddOptionDouble(&options->delaunay_meshing->distance_sigma_factor,
                    "distance_sigma_factor",
                    0);
    AddOptionDouble(&options->delaunay_meshing->quality_regularization,
                    "quality_regularization",
                    0);
    AddOptionDouble(&options->delaunay_meshing->max_side_length_factor,
                    "max_side_length_factor",
                    0);
    AddOptionDouble(&options->delaunay_meshing->max_side_length_percentile,
                    "max_side_length_percentile",
                    0);
    AddOptionInt(&options->delaunay_meshing->num_threads, "num_threads", -1);
  }
};
#endif  // COLMAP_MVS_ENABLED

// Read the specified reference image names from a patch match configuration.
std::vector<std::pair<std::string, std::string>> ReadPatchMatchConfig(
    const std::filesystem::path& config_path) {
  std::ifstream file(config_path);
  THROW_CHECK_FILE_OPEN(file, config_path);

  std::string line;
  std::string ref_image_name;
  std::vector<std::pair<std::string, std::string>> images;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = line;
    } else {
      images.emplace_back(ref_image_name, line);
      ref_image_name.clear();
    }
  }

  return images;
}

}  // namespace

DenseReconstructionOptionsWidget::DenseReconstructionOptionsWidget(
    QWidget* parent, OptionManager* options)
    : QWidget(parent) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Dense reconstruction options");

  QGridLayout* grid = new QGridLayout(this);

  QTabWidget* tab_widget = new QTabWidget(this);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
#if defined(COLMAP_MVS_ENABLED)
  tab_widget->addTab(new StereoOptionsTab(this, options), "Stereo");
  tab_widget->addTab(new FusionOptionsTab(this, options), "Fusion");
  tab_widget->addTab(new MeshingOptionsTab(this, options), "Meshing");
#endif

  grid->addWidget(tab_widget, 0, 0);
}

DenseReconstructionWidget::DenseReconstructionWidget(MainWindow* main_window,
                                                     OptionManager* options)
    : QWidget(main_window),
      main_window_(main_window),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)),
      options_widget_(new DenseReconstructionOptionsWidget(this, options)),
      photometric_done_(false),
      geometric_done_(false) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Dense reconstruction");
  resize(main_window_->size().width() - 20, main_window_->size().height() - 20);

  QGridLayout* grid = new QGridLayout(this);

  undistortion_button_ = new QPushButton(tr("Undistortion"), this);
  connect(undistortion_button_,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::Undistort);
  grid->addWidget(undistortion_button_, 0, 0, Qt::AlignLeft);

  stereo_button_ = new QPushButton(tr("Stereo"), this);
  connect(stereo_button_,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::Stereo);
  grid->addWidget(stereo_button_, 0, 1, Qt::AlignLeft);

  fusion_button_ = new QPushButton(tr("Fusion"), this);
  connect(fusion_button_,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::Fusion);
  grid->addWidget(fusion_button_, 0, 2, Qt::AlignLeft);

  poisson_meshing_button_ = new QPushButton(tr("Poisson"), this);
  connect(poisson_meshing_button_,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::PoissonMeshing);
  grid->addWidget(poisson_meshing_button_, 0, 3, Qt::AlignLeft);

  delaunay_meshing_button_ = new QPushButton(tr("Delaunay"), this);
  connect(delaunay_meshing_button_,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::DelaunayMeshing);
  grid->addWidget(delaunay_meshing_button_, 0, 4, Qt::AlignLeft);

  QPushButton* options_button = new QPushButton(tr("Options"), this);
  connect(options_button,
          &QPushButton::released,
          options_widget_,
          &OptionsWidget::show);
  grid->addWidget(options_button, 0, 5, Qt::AlignLeft);

  QLabel* workspace_path_label = new QLabel("Workspace", this);
  grid->addWidget(workspace_path_label, 0, 6, Qt::AlignRight);

  workspace_path_text_ = new QLineEdit(this);
  grid->addWidget(workspace_path_text_, 0, 7, Qt::AlignRight);
  connect(workspace_path_text_,
          &QLineEdit::textChanged,
          this,
          &DenseReconstructionWidget::RefreshWorkspace,
          Qt::QueuedConnection);

  QPushButton* refresh_path_button = new QPushButton(tr("Refresh"), this);
  connect(refresh_path_button,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::RefreshWorkspace,
          Qt::QueuedConnection);
  grid->addWidget(refresh_path_button, 0, 8, Qt::AlignRight);

  QPushButton* workspace_path_button = new QPushButton(tr("Select"), this);
  connect(workspace_path_button,
          &QPushButton::released,
          this,
          &DenseReconstructionWidget::SelectWorkspacePath,
          Qt::QueuedConnection);
  grid->addWidget(workspace_path_button, 0, 9, Qt::AlignRight);

  QStringList table_header;
  table_header << "image_name"
               << ""
               << "photometric"
               << "geometric"
               << "src_images";

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(table_header.size());
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->verticalHeader()->setDefaultSectionSize(25);

  grid->addWidget(table_widget_, 1, 0, 1, 10);

  grid->setColumnStretch(4, 1);

  image_viewer_widget_ = new ImageViewerWidget(this);
  image_viewer_widget_->setWindowFlags(
      Qt::Dialog | Qt::WindowTitleHint | Qt::WindowMinimizeButtonHint |
      Qt::WindowMaximizeButtonHint | Qt::WindowCloseButtonHint);

  refresh_workspace_action_ = new QAction(this);
  connect(refresh_workspace_action_,
          &QAction::triggered,
          this,
          &DenseReconstructionWidget::RefreshWorkspace);

  write_fused_points_action_ = new QAction(this);
  connect(write_fused_points_action_,
          &QAction::triggered,
          this,
          &DenseReconstructionWidget::WriteFusedPoints);

  write_surface_mesh_action_ = new QAction(this);
  connect(write_surface_mesh_action_,
          &QAction::triggered,
          this,
          &DenseReconstructionWidget::WriteSurfaceMesh);

  RefreshWorkspace();
}

void DenseReconstructionWidget::showEvent(QShowEvent* event) {
  RefreshWorkspace();
}

void DenseReconstructionWidget::Show(
    std::shared_ptr<const Reconstruction> reconstruction) {
  reconstruction_ = std::move(reconstruction);
  show();
  raise();
}

void DenseReconstructionWidget::Undistort() {
  const auto workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  if (reconstruction_ == nullptr || reconstruction_->NumRegFrames() < 2) {
    QMessageBox::critical(
        this, "", tr("No reconstruction selected in main window"));
    return;
  }

  auto undistorter = std::make_unique<ControllerThread<COLMAPUndistorter>>(
      std::make_shared<COLMAPUndistorter>(COLMAPUndistorter::Options(),
                                          UndistortCameraOptions(),
                                          *reconstruction_,
                                          *options_->image_path,
                                          workspace_path));
  undistorter->AddCallback(Thread::FINISHED_CALLBACK,
                           [this]() { refresh_workspace_action_->trigger(); });
  thread_control_widget_->StartThread(
      "Undistorting...", true, std::move(undistorter));
}

void DenseReconstructionWidget::Stereo() {
  const auto workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

#if defined(COLMAP_MVS_ENABLED) && defined(COLMAP_CUDA_ENABLED)
  auto processor =
      std::make_unique<ControllerThread<mvs::PatchMatchController>>(
          std::make_shared<mvs::PatchMatchController>(
              *options_->patch_match_stereo, workspace_path, "COLMAP", ""));
  processor->AddCallback(Thread::FINISHED_CALLBACK,
                         [this]() { refresh_workspace_action_->trigger(); });
  thread_control_widget_->StartThread("Stereo...", true, std::move(processor));
#elif !defined(COLMAP_MVS_ENABLED)
  QMessageBox::critical(this,
                        "",
                        tr("Dense stereo reconstruction requires the MVS "
                           "module, which is not available in this build."));
#else
  QMessageBox::critical(this,
                        "",
                        tr("Dense stereo reconstruction requires CUDA, which "
                           "is not available on your system."));
#endif
}

void DenseReconstructionWidget::Fusion() {
#if !defined(COLMAP_MVS_ENABLED)
  QMessageBox::critical(this,
                        "",
                        tr("Stereo fusion requires the MVS module, which "
                           "is not available in this build."));
#else
  const auto workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  std::string input_type;
  if (geometric_done_) {
    input_type = "geometric";
  } else if (photometric_done_) {
    input_type = "photometric";
  } else {
    QMessageBox::critical(
        this, "", tr("All images must be processed prior to fusion"));
  }

  auto fuser = std::make_unique<ControllerThread<mvs::StereoFusion>>(
      std::make_shared<mvs::StereoFusion>(
          *options_->stereo_fusion, workspace_path, "COLMAP", "", input_type));
  fuser->AddCallback(
      Thread::FINISHED_CALLBACK, [this, fuser = fuser.get(), workspace_path]() {
        auto fused_points = fuser->GetController()->GetFusedPoints();
        auto fused_points_visibility =
            fuser->GetController()->GetFusedPointsVisibility();
        const auto output_path = workspace_path / kFusedFileName;
        WriteBinaryPlyPoints(output_path, fused_points);
        mvs::WritePointsVisibility(AddFileExtension(output_path, ".vis"),
                                   fused_points_visibility);
        main_window_->model_viewer_widget_->point_cloud =
            std::move(fused_points);
        write_fused_points_action_->trigger();
      });
  thread_control_widget_->StartThread("Fusion...", true, std::move(fuser));
#endif
}

void DenseReconstructionWidget::LoadAndDisplayMesh(
    const std::filesystem::path& mesh_path) {
  try {
    main_window_->model_viewer_widget_->surface_mesh = ReadPlyMesh(mesh_path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to read surface mesh: " << e.what();
  }
  write_surface_mesh_action_->trigger();
}

void DenseReconstructionWidget::PoissonMeshing() {
#if !defined(COLMAP_MVS_ENABLED)
  QMessageBox::critical(this,
                        "",
                        tr("Poisson meshing requires the MVS module, which "
                           "is not available in this build."));
#else
  const auto workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  if (ExistsFile(workspace_path / kFusedFileName)) {
    thread_control_widget_->StartFunction(
        "Poisson Meshing...", [this, workspace_path]() {
          mvs::PoissonMeshing(*options_->poisson_meshing,
                              workspace_path / kFusedFileName,
                              workspace_path / kPoissonMeshedFileName);
          LoadAndDisplayMesh(workspace_path / kPoissonMeshedFileName);
        });
  }
#endif
}

void DenseReconstructionWidget::DelaunayMeshing() {
#if !defined(COLMAP_MVS_ENABLED)
  QMessageBox::critical(this,
                        "",
                        tr("Delaunay meshing requires the MVS module, which "
                           "is not available in this build."));
#elif defined(COLMAP_CGAL_ENABLED)
  const auto workspace_path = GetWorkspacePath();
  if (workspace_path.empty()) {
    return;
  }

  if (ExistsFile(workspace_path / kFusedFileName)) {
    thread_control_widget_->StartFunction(
        "Delaunay Meshing...", [this, workspace_path]() {
          mvs::DenseDelaunayMeshing(*options_->delaunay_meshing,
                                    workspace_path,
                                    workspace_path / kDelaunayMeshedFileName);
          LoadAndDisplayMesh(workspace_path / kDelaunayMeshedFileName);
        });
  }
#else
  QMessageBox::critical(this,
                        "",
                        tr("Delaunay meshing requires CGAL, which "
                           "is not available on your system."));
#endif
}

void DenseReconstructionWidget::SelectWorkspacePath() {
  std::string workspace_path;
  if (workspace_path_text_->text().isEmpty()) {
    workspace_path = GetParentDir(*options_->project_path).string();
  } else {
    workspace_path = workspace_path_text_->text().toUtf8().constData();
  }

  workspace_path_text_->setText(
      QFileDialog::getExistingDirectory(this,
                                        tr("Select workspace path..."),
                                        QString::fromStdString(workspace_path),
                                        QFileDialog::ShowDirsOnly));

  RefreshWorkspace();
}

std::filesystem::path DenseReconstructionWidget::GetWorkspacePath() {
  std::filesystem::path workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (ExistsDir(workspace_path)) {
    return workspace_path;
  } else {
    QMessageBox::critical(this, "", tr("Invalid workspace path"));
    return "";
  }
}

void DenseReconstructionWidget::RefreshWorkspace() {
  table_widget_->clearContents();
  table_widget_->setRowCount(0);

  const std::filesystem::path workspace_path =
      workspace_path_text_->text().toUtf8().constData();
  if (ExistsDir(workspace_path)) {
    undistortion_button_->setEnabled(true);
  } else {
    undistortion_button_->setEnabled(false);
    stereo_button_->setEnabled(false);
    fusion_button_->setEnabled(false);
    poisson_meshing_button_->setEnabled(false);
    delaunay_meshing_button_->setEnabled(false);
    return;
  }

  images_path_ = workspace_path / "images";
  depth_maps_path_ = workspace_path / "stereo/depth_maps";
  normal_maps_path_ = workspace_path / "stereo/normal_maps";
  const auto config_path = workspace_path / "stereo/patch-match.cfg";

  if (ExistsDir(images_path_) && ExistsDir(depth_maps_path_) &&
      ExistsDir(normal_maps_path_) && ExistsDir(workspace_path / "sparse") &&
      ExistsDir(workspace_path / "stereo/consistency_graphs") &&
      ExistsFile(config_path)) {
    stereo_button_->setEnabled(true);
  } else {
    stereo_button_->setEnabled(false);
    fusion_button_->setEnabled(false);
    poisson_meshing_button_->setEnabled(false);
    delaunay_meshing_button_->setEnabled(false);
    return;
  }

  const auto images = ReadPatchMatchConfig(config_path);
  table_widget_->setRowCount(images.size());

  for (size_t i = 0; i < images.size(); ++i) {
    const std::string image_name = images[i].first;
    const std::string src_images = images[i].second;
    const auto image_path = images_path_ / image_name;

    QTableWidgetItem* image_name_item =
        new QTableWidgetItem(QString::fromStdString(image_name));
    table_widget_->setItem(i, 0, image_name_item);

    QPushButton* image_button = new QPushButton("Image");
    connect(
        image_button, &QPushButton::released, [this, image_name, image_path]() {
          image_viewer_widget_->setWindowTitle(
              QString("Image for %1").arg(image_name.c_str()));
          image_viewer_widget_->ReadAndShow(image_path);
        });
    table_widget_->setCellWidget(i, 1, image_button);

    table_widget_->setCellWidget(
        i, 2, GenerateTableButtonWidget(image_name, "photometric"));
    table_widget_->setCellWidget(
        i, 3, GenerateTableButtonWidget(image_name, "geometric"));

    QTableWidgetItem* src_images_item =
        new QTableWidgetItem(QString::fromStdString(src_images));
    table_widget_->setItem(i, 4, src_images_item);
  }

  table_widget_->resizeColumnsToContents();

  fusion_button_->setEnabled(photometric_done_ || geometric_done_);
  poisson_meshing_button_->setEnabled(
      ExistsFile(workspace_path / kFusedFileName));
  delaunay_meshing_button_->setEnabled(
      ExistsFile(workspace_path / kFusedFileName));
}

void DenseReconstructionWidget::WriteFusedPoints() {
  poisson_meshing_button_->setEnabled(true);
  delaunay_meshing_button_->setEnabled(true);
  main_window_->RenderNow();
}

void DenseReconstructionWidget::WriteSurfaceMesh() {
  main_window_->RenderNow();
}

QWidget* DenseReconstructionWidget::GenerateTableButtonWidget(
    const std::string& image_name, const std::string& type) {
  THROW_CHECK(type == "photometric" || type == "geometric");
  const bool photometric = type == "photometric";

  if (photometric) {
    photometric_done_ = true;
  } else {
    geometric_done_ = true;
  }

  const auto depth_map_path =
      depth_maps_path_ /
      StringPrintf("%s.%s.bin", image_name.c_str(), type.c_str());
  const auto normal_map_path =
      normal_maps_path_ /
      StringPrintf("%s.%s.bin", image_name.c_str(), type.c_str());

  QWidget* button_widget = new QWidget();
  QGridLayout* button_layout = new QGridLayout(button_widget);
  button_layout->setContentsMargins(0, 0, 0, 0);

  QPushButton* depth_map_button = new QPushButton("Depth map", button_widget);
  if (ExistsFile(depth_map_path)) {
    connect(depth_map_button,
            &QPushButton::released,
            [this, image_name, depth_map_path]() {
#if defined(COLMAP_MVS_ENABLED)
              mvs::DepthMap depth_map;
              depth_map.Read(depth_map_path);
              image_viewer_widget_->setWindowTitle(
                  QString("Depth map for %1").arg(image_name.c_str()));
              image_viewer_widget_->ShowBitmap(depth_map.ToBitmap(2, 98));
#endif
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
  if (ExistsFile(normal_map_path)) {
    connect(normal_map_button,
            &QPushButton::released,
            [this, image_name, normal_map_path]() {
#if defined(COLMAP_MVS_ENABLED)
              mvs::NormalMap normal_map;
              normal_map.Read(normal_map_path);
              image_viewer_widget_->setWindowTitle(
                  QString("Normal map for %1").arg(image_name.c_str()));
              image_viewer_widget_->ShowBitmap(normal_map.ToBitmap());
#endif
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
