// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/automatic_reconstruction_widget.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/ui/main_window.h"

namespace colmap {

AutomaticReconstructionWidget::AutomaticReconstructionWidget(
    MainWindow* main_window)
    : OptionsWidget(main_window),
      main_window_(main_window),
      thread_control_widget_(new ThreadControlWidget(this)) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Automatic reconstruction");

  AddOptionDirPath(&options_.workspace_path, "Workspace folder");
  AddSpacer();
  AddOptionDirPath(&options_.image_path, "Image folder");
  AddSpacer();
  AddOptionDirPath(&options_.mask_path, "Mask folder");
  AddSpacer();
  AddOptionFilePath(&options_.vocab_tree_path, "Vocabulary tree<br>(optional)");

  AddSpacer();

  data_type_cb_ = new QComboBox(this);
  data_type_cb_->addItem("Individual images");
  data_type_cb_->addItem("Video frames");
  data_type_cb_->addItem("Internet images");
  AddWidgetRow("Data type", data_type_cb_);

  quality_cb_ = new QComboBox(this);
  quality_cb_->addItem("Low");
  quality_cb_->addItem("Medium");
  quality_cb_->addItem("High");
  quality_cb_->addItem("Extreme");
  quality_cb_->setCurrentIndex(2);
  AddWidgetRow("Quality", quality_cb_);

  AddSpacer();

  AddOptionBool(&options_.single_camera, "Shared intrinsics");
  AddOptionBool(&options_.single_camera_per_folder,
                "Shared intrinsics per sub-folder");
  AddOptionBool(&options_.sparse, "Sparse model");
  AddOptionBool(&options_.dense, "Dense model");

  mesher_cb_ = new QComboBox(this);
  mesher_cb_->addItem("Poisson");
  mesher_cb_->addItem("Delaunay");
  mesher_cb_->setCurrentIndex(0);
  AddWidgetRow("Mesher", mesher_cb_);

  AddSpacer();

  AddOptionInt(&options_.num_threads, "num_threads", -1);
  AddOptionInt(&options_.random_seed, "random_seed", -1);
  AddOptionBool(&options_.use_gpu, "GPU");
  AddOptionText(&options_.gpu_index, "gpu_index");

#ifdef CASPAR_ENABLED
  AddSpacer();
  AddSection("Bundle Adjustment Backend");

  ba_backend_cb_ = new QComboBox(this);
  ba_backend_cb_->addItem("CERES");
  ba_backend_cb_->addItem("CASPAR");
  ba_backend_cb_->setCurrentIndex(static_cast<int>(options_.ba_backend));
  AddWidgetRow("Backend", ba_backend_cb_);
#endif

  AddSpacer();

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
  connect(run_button,
          &QPushButton::released,
          this,
          &AutomaticReconstructionWidget::Run);

  render_result_ = new QAction(this);
  connect(render_result_,
          &QAction::triggered,
          this,
          &AutomaticReconstructionWidget::RenderResult,
          Qt::QueuedConnection);
}

void AutomaticReconstructionWidget::Run() {
  WriteOptions();

  if (!ExistsDir(options_.workspace_path)) {
    QMessageBox::critical(this, "", tr("Invalid workspace folder"));
    return;
  }

  if (!ExistsDir(options_.image_path)) {
    QMessageBox::critical(this, "", tr("Invalid image folder"));
    return;
  }

  switch (data_type_cb_->currentIndex()) {
    case 0:
      options_.data_type =
          AutomaticReconstructionController::DataType::INDIVIDUAL;
      break;
    case 1:
      options_.data_type = AutomaticReconstructionController::DataType::VIDEO;
      break;
    case 2:
      options_.data_type =
          AutomaticReconstructionController::DataType::INTERNET;
      break;
    default:
      options_.data_type =
          AutomaticReconstructionController::DataType::INDIVIDUAL;
      break;
  }

  switch (quality_cb_->currentIndex()) {
    case 0:
      options_.quality = AutomaticReconstructionController::Quality::LOW;
      break;
    case 1:
      options_.quality = AutomaticReconstructionController::Quality::MEDIUM;
      break;
    case 2:
      options_.quality = AutomaticReconstructionController::Quality::HIGH;
      break;
    case 3:
      options_.quality = AutomaticReconstructionController::Quality::EXTREME;
      break;
    default:
      options_.quality = AutomaticReconstructionController::Quality::HIGH;
      break;
  }

  switch (mesher_cb_->currentIndex()) {
    case 0:
      options_.mesher = AutomaticReconstructionController::Mesher::POISSON;
      break;
    case 1:
      options_.mesher = AutomaticReconstructionController::Mesher::DELAUNAY;
      break;
    default:
      options_.mesher = AutomaticReconstructionController::Mesher::POISSON;
      break;
  }

#ifdef CASPAR_ENABLED
  options_.ba_backend =
      static_cast<BundleAdjustmentBackend>(ba_backend_cb_->currentIndex());
#endif

  main_window_->reconstruction_manager_->Clear();
  main_window_->reconstruction_manager_widget_->Update();
  main_window_->RenderClear();
  main_window_->RenderNow();

  auto controller = std::make_unique<AutomaticReconstructionController>(
      options_, main_window_->reconstruction_manager_);
  controller->AddCallback(Thread::FINISHED_CALLBACK,
                          [this]() { render_result_->trigger(); });
  controller->Setup();
  thread_control_widget_->StartThread(
      "Reconstructing...", true, std::move(controller));
}

void AutomaticReconstructionWidget::RenderResult() {
  if (main_window_->reconstruction_manager_->Size() > 0) {
    main_window_->reconstruction_manager_widget_->Update();
    main_window_->RenderClear();
    main_window_->RenderNow();
  }

  if (options_.sparse) {
    QMessageBox::information(
        this,
        "",
        tr("Imported the reconstructed sparse models for visualization. The "
           "models were also exported to the <i>sparse</i> sub-folder in the "
           "workspace."));
  }

  if (options_.dense) {
    const auto dense_path = options_.workspace_path / "dense" / "0";
    std::filesystem::path meshing_path;
    if (options_.mesher == AutomaticReconstructionController::Mesher::POISSON) {
      meshing_path = dense_path / "meshed-poisson.ply";
    } else {
      meshing_path = dense_path / "meshed-delaunay.ply";
    }

    if (ExistsFile(meshing_path)) {
      try {
        main_window_->model_viewer_widget_->surface_mesh =
            ReadPlyMesh(meshing_path);
        main_window_->RenderNow();
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to read surface mesh: " << e.what();
      }
    }

    QMessageBox::information(
        this,
        "",
        tr("To visualize the reconstructed dense point cloud or surface mesh, "
           "navigate to the <i>dense</i> sub-folder in your workspace with "
           "<i>File > Import...</i>."));
  }
}

}  // namespace colmap
