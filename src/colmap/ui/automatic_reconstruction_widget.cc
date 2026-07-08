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

  // Retrieval type selector.
  QLabel* retrieval_label =
      new QLabel(tr("Image retrieval"), this);
  retrieval_label->setFont(font());
  retrieval_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(retrieval_label, grid_layout_->rowCount(), 0);

  retrieval_type_cb_ = new QComboBox(this);
  retrieval_type_cb_->addItem("None (exhaustive matching)");
  retrieval_type_cb_->addItem("Vocabulary Tree");
  retrieval_type_cb_->addItem("MixVPR (global descriptor)");
  grid_layout_->addWidget(retrieval_type_cb_,
                          grid_layout_->rowCount() - 1, 1);

  // Vocab tree path row (visible when Vocabulary Tree selected).
  {
    vocab_tree_row_ = new QWidget(this);
    QHBoxLayout* hbox = new QHBoxLayout(vocab_tree_row_);
    hbox->setContentsMargins(0, 0, 0, 0);
    QLabel* label = new QLabel(tr("Vocab tree path"), vocab_tree_row_);
    QLineEdit* edit = new QLineEdit(vocab_tree_row_);
    edit->setText(QString::fromStdString(options_.vocab_tree_path.string()));
    QPushButton* btn = new QPushButton(tr("Select file"), vocab_tree_row_);
    hbox->addWidget(label);
    hbox->addWidget(edit);
    hbox->addWidget(btn);
    AddWidgetRow("", vocab_tree_row_);
    connect(btn, &QPushButton::released, this, [edit]() {
      edit->setText(QFileDialog::getOpenFileName(edit, tr("Select file")));
    });
    edit->setObjectName("auto_vocab_tree_edit");
  }

  // Global descriptor model row (visible when MixVPR selected).
  {
    global_descriptor_row_ = new QWidget(this);
    QHBoxLayout* hbox = new QHBoxLayout(global_descriptor_row_);
    hbox->setContentsMargins(0, 0, 0, 0);
    QLabel* label =
        new QLabel(tr("MixVPR model (ONNX)"), global_descriptor_row_);
    QLineEdit* edit = new QLineEdit(global_descriptor_row_);
    edit->setText(
        QString::fromStdString(options_.global_descriptor_path.string()));
    QPushButton* btn =
        new QPushButton(tr("Select file"), global_descriptor_row_);
    hbox->addWidget(label);
    hbox->addWidget(edit);
    hbox->addWidget(btn);
    AddWidgetRow("", global_descriptor_row_);
    connect(btn, &QPushButton::released, this, [edit]() {
      edit->setText(QFileDialog::getOpenFileName(edit, tr("Select file")));
    });
    edit->setObjectName("auto_global_descriptor_edit");
  }

  connect(retrieval_type_cb_,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          this,
          &AutomaticReconstructionWidget::UpdateRetrievalFields);

  AddSpacer();

  QLabel* data_type_label = new QLabel(tr("Data type"), this);
  data_type_label->setFont(font());
  data_type_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(data_type_label, grid_layout_->rowCount(), 0);

  data_type_cb_ = new QComboBox(this);
  data_type_cb_->addItem("Individual images");
  data_type_cb_->addItem("Video frames");
  data_type_cb_->addItem("Internet images");
  grid_layout_->addWidget(data_type_cb_, grid_layout_->rowCount() - 1, 1);

  QLabel* quality_label = new QLabel(tr("Quality"), this);
  quality_label->setFont(font());
  quality_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(quality_label, grid_layout_->rowCount(), 0);

  quality_cb_ = new QComboBox(this);
  quality_cb_->addItem("Low");
  quality_cb_->addItem("Medium");
  quality_cb_->addItem("High");
  quality_cb_->addItem("Extreme");
  quality_cb_->setCurrentIndex(2);
  grid_layout_->addWidget(quality_cb_, grid_layout_->rowCount() - 1, 1);

  AddSpacer();

  AddOptionBool(&options_.single_camera, "Shared intrinsics");
  AddOptionBool(&options_.single_camera_per_folder,
                "Shared intrinsics per sub-folder");
  AddOptionBool(&options_.sparse, "Sparse model");
  AddOptionBool(&options_.dense, "Dense model");

  QLabel* mesher_label = new QLabel(tr("Mesher"), this);
  mesher_label->setFont(font());
  mesher_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(mesher_label, grid_layout_->rowCount(), 0);

  mesher_cb_ = new QComboBox(this);
  mesher_cb_->addItem("Poisson");
  mesher_cb_->addItem("Delaunay");
  mesher_cb_->setCurrentIndex(0);
  grid_layout_->addWidget(mesher_cb_, grid_layout_->rowCount() - 1, 1);

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

void AutomaticReconstructionWidget::UpdateRetrievalFields() {
  const int idx = retrieval_type_cb_->currentIndex();
  // 0 = None, 1 = Vocabulary Tree, 2 = MixVPR
  if (vocab_tree_row_)
    vocab_tree_row_->setVisible(idx == 1);
  if (global_descriptor_row_)
    global_descriptor_row_->setVisible(idx == 2);
}

void AutomaticReconstructionWidget::Run() {
  WriteOptions();

  // Sync custom retrieval widgets (not managed by OptionsWidget::WriteOptions).
  const int retrieval_idx = retrieval_type_cb_->currentIndex();
  if (retrieval_idx == 1) {
    // Vocabulary Tree.
    QLineEdit* edit =
        vocab_tree_row_
            ? vocab_tree_row_->findChild<QLineEdit*>("auto_vocab_tree_edit")
            : nullptr;
    if (edit) options_.vocab_tree_path = edit->text().toStdString();
    options_.global_descriptor_path.clear();
  } else if (retrieval_idx == 2) {
    // MixVPR global descriptor.
    QLineEdit* edit =
        global_descriptor_row_
            ? global_descriptor_row_->findChild<QLineEdit*>(
                  "auto_global_descriptor_edit")
            : nullptr;
    if (edit) options_.global_descriptor_path = edit->text().toStdString();
    options_.vocab_tree_path.clear();
  } else {
    // None: clear both.
    options_.vocab_tree_path.clear();
    options_.global_descriptor_path.clear();
  }

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
