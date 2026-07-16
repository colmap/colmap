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

#include "colmap/ui/feature_matching_widget.h"

#include "colmap/controllers/feature_matching.h"
#include "colmap/feature/sift.h"
#include "colmap/retrieval/global_descriptor_model.h"
#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/aliked.h"
#endif
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"
#include "colmap/util/file.h"

#include <filesystem>

namespace colmap {

class FeatureMatchingTab : public QWidget {
 public:
  FeatureMatchingTab(QWidget* parent, OptionManager* options);

  virtual void Run() = 0;

 private:
  void showEvent(QShowEvent* event);

 protected:
  void ReadOptions();
  void WriteOptions();
  void CreateGeneralOptions();

  OptionManager* options_;
  OptionsWidget* options_widget_;
  QGridLayout* grid_layout_;
  ThreadControlWidget* thread_control_widget_;
  QComboBox* matcher_type_cb_;
  std::vector<FeatureMatcherType> matcher_types_;
};

class ExhaustiveMatchingTab : public FeatureMatchingTab {
 public:
  ExhaustiveMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;
};

class SequentialMatchingTab : public FeatureMatchingTab {
 public:
  SequentialMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;

 private:
  void UpdateLoopDetectionFields();
  QComboBox* loop_detection_type_cb_;
  // Rows added via AddOptionFilePath; we track the grid row of the lineEdit
  // so we can hide the label (row, col 0), lineEdit (row, col 1), and
  // button (row+1, col 1) together.
  int vocab_tree_grid_row_ = -1;
  int global_descriptor_grid_row_ = -1;
};

class VocabTreeMatchingTab : public FeatureMatchingTab {
 public:
  VocabTreeMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;
};

#ifdef COLMAP_ONNX_ENABLED
class GlobalDescriptorMatchingTab : public FeatureMatchingTab {
 public:
  GlobalDescriptorMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;

 private:
  void OnModelTypeChanged(int index);
  QComboBox* model_type_cb_;
  QLineEdit* model_path_edit_ = nullptr;
};
#endif  // COLMAP_ONNX_ENABLED

class SpatialMatchingTab : public FeatureMatchingTab {
 public:
  SpatialMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;
};

class TransitiveMatchingTab : public FeatureMatchingTab {
 public:
  TransitiveMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;
};

class CustomMatchingTab : public FeatureMatchingTab {
 public:
  CustomMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;

 private:
  std::filesystem::path custom_match_list_path_;
  QComboBox* custom_match_type_cb_;
};

FeatureMatchingTab::FeatureMatchingTab(QWidget* parent, OptionManager* options)
    : QWidget(parent),
      options_(options),
      options_widget_(new OptionsWidget(this)),
      grid_layout_(new QGridLayout(this)),
      thread_control_widget_(new ThreadControlWidget(this)) {}

void FeatureMatchingTab::CreateGeneralOptions() {
  options_widget_->AddSpacer();
  options_widget_->AddSection("Shared options");

  matcher_type_cb_ = new QComboBox(options_widget_);
  auto add_matcher_type = [this](FeatureMatcherType type) {
    matcher_type_cb_->addItem(
        QString::fromStdString(std::string(FeatureMatcherTypeToString(type))));
    matcher_types_.push_back(type);
  };
  add_matcher_type(FeatureMatcherType::SIFT_BRUTEFORCE);
#ifdef COLMAP_ONNX_ENABLED
  add_matcher_type(FeatureMatcherType::SIFT_LIGHTGLUE);
  add_matcher_type(FeatureMatcherType::ALIKED_BRUTEFORCE);
  add_matcher_type(FeatureMatcherType::ALIKED_LIGHTGLUE);
#endif
  options_widget_->AddWidgetRow("Type", matcher_type_cb_);

  options_widget_->AddOptionInt(
      &options_->feature_matching->num_threads, "num_threads", -1);
  options_widget_->AddOptionBool(&options_->feature_matching->use_gpu,
                                 "use_gpu");
  options_widget_->AddOptionText(&options_->feature_matching->gpu_index,
                                 "gpu_index");
  options_widget_->AddOptionInt(&options_->feature_matching->max_num_matches,
                                "max_num_matches");
  options_widget_->AddOptionBool(&options_->feature_matching->guided_matching,
                                 "guided_matching");
  options_widget_->AddOptionBool(
      &options_->feature_matching->skip_geometric_verification,
      "skip_geometric_verification");
  options_widget_->AddOptionBool(&options_->feature_matching->rig_verification,
                                 "rig_verification");
  options_widget_->AddOptionBool(
      &options_->feature_matching->skip_image_pairs_in_same_frame,
      "skip_image_pairs_in_same_frame");

  options_widget_->AddOptionDouble(&options_->feature_matching->sift->max_ratio,
                                   "sift.max_ratio");
  options_widget_->AddOptionDouble(
      &options_->feature_matching->sift->max_distance, "sift.max_distance");
  options_widget_->AddOptionBool(&options_->feature_matching->sift->cross_check,
                                 "sift.cross_check");

  options_widget_->AddSpacer();
  options_widget_->AddSection("Geometric verification");

  options_widget_->AddOptionDouble(
      &options_->two_view_geometry->ransac_options.max_error, "max_error");
  options_widget_->AddOptionDouble(
      &options_->two_view_geometry->ransac_options.confidence,
      "confidence",
      0,
      1,
      0.00001,
      5);
  options_widget_->AddOptionInt(
      &options_->two_view_geometry->ransac_options.max_num_trials,
      "max_num_trials");
  options_widget_->AddOptionDouble(
      &options_->two_view_geometry->ransac_options.min_inlier_ratio,
      "min_inlier_ratio",
      0,
      1,
      0.001,
      3);
  options_widget_->AddOptionInt(
      &options_->two_view_geometry->ransac_options.random_seed, "random_seed");
  options_widget_->AddOptionInt(&options_->two_view_geometry->min_num_inliers,
                                "min_num_inliers");
  options_widget_->AddOptionBool(&options_->two_view_geometry->multiple_models,
                                 "multiple_models");
  options_widget_->AddOptionBool(&options_->two_view_geometry->detect_watermark,
                                 "detect_watermarks");
  options_widget_->AddOptionBool(
      &options_->two_view_geometry->filter_stationary_matches,
      "filter_stationary_matches");
  options_widget_->AddSpacer();

  QScrollArea* options_scroll_area = new QScrollArea(this);
  options_scroll_area->setAlignment(Qt::AlignHCenter);
  options_scroll_area->setWidget(options_widget_);
  grid_layout_->addWidget(options_scroll_area, grid_layout_->rowCount(), 0);

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 0);
  connect(run_button, &QPushButton::released, this, &FeatureMatchingTab::Run);
}

void FeatureMatchingTab::showEvent(QShowEvent* event) { ReadOptions(); }

void FeatureMatchingTab::ReadOptions() {
  for (size_t i = 0; i < matcher_types_.size(); ++i) {
    if (options_->feature_matching->type == matcher_types_[i]) {
      matcher_type_cb_->setCurrentIndex(i);
      break;
    }
  }
  options_widget_->ReadOptions();
}

void FeatureMatchingTab::WriteOptions() {
  options_widget_->WriteOptions();
  options_->feature_matching->type =
      matcher_types_[matcher_type_cb_->currentIndex()];
}

ExhaustiveMatchingTab::ExhaustiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(
      &options_->exhaustive_pairing->block_size, "block_size", 2);

  CreateGeneralOptions();
}

void ExhaustiveMatchingTab::Run() {
  WriteOptions();

  auto matcher = CreateExhaustiveFeatureMatcher(*options_->exhaustive_pairing,
                                                *options_->feature_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

SequentialMatchingTab::SequentialMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options_->sequential_pairing->overlap,
                                "overlap");
  options_widget_->AddOptionBool(
      &options_->sequential_pairing->quadratic_overlap, "quadratic_overlap");

  // Loop detection type selector.
  loop_detection_type_cb_ = new QComboBox(this);
  loop_detection_type_cb_->addItem("None");             // 0
  loop_detection_type_cb_->addItem("Vocabulary Tree");  // 1
#ifdef COLMAP_ONNX_ENABLED
  loop_detection_type_cb_->addItem("MixVPR");           // 2
  loop_detection_type_cb_->addItem("MegaLoc");           // 3
#endif
  options_widget_->AddWidgetRow("Loop detection", loop_detection_type_cb_);

  // Use standard AddOptionFilePath for both path types — identical height,
  // no layout jump when swapping.  We track the grid row of each lineEdit
  // so we can hide the entire row (label + lineEdit + button) together
  // via the QGridLayout.
  if (QGridLayout* grid = options_widget_->findChild<QGridLayout*>()) {
    // Vocab tree path.
    vocab_tree_grid_row_ = grid->rowCount();
    options_widget_->AddOptionFilePath(
        &options_->sequential_pairing->vocab_tree_path, "Vocab tree path");

    // MixVPR/MegaLoc model path (ONNX only).
#ifdef COLMAP_ONNX_ENABLED
    global_descriptor_grid_row_ = grid->rowCount();
    options_widget_->AddOptionFilePath(
        &options_->sequential_pairing->loop_detection_model_path,
        "Model path (ONNX, optional)");
#endif
  }

  // General loop detection parameters (visible when any type is selected).
  options_widget_->AddOptionInt(
      &options_->sequential_pairing->loop_detection_period,
      "loop_detection_period");
  options_widget_->AddOptionInt(
      &options_->sequential_pairing->loop_detection_num_images,
      "loop_detection_num_images");
  options_widget_->AddOptionInt(
      &options_->sequential_pairing->loop_detection_num_nearest_neighbors,
      "loop_detection_num_nearest_neighbors");
  options_widget_->AddOptionInt(
      &options_->sequential_pairing->loop_detection_num_checks,
      "loop_detection_num_checks",
      1);
  options_widget_->AddOptionInt(
      &options_->sequential_pairing
           ->loop_detection_num_images_after_verification,
      "loop_detection_num_images_after_verification",
      0);
  options_widget_->AddOptionInt(
      &options_->sequential_pairing->loop_detection_max_num_features,
      "loop_detection_max_num_features",
      -1);

  connect(loop_detection_type_cb_,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          this,
          &SequentialMatchingTab::UpdateLoopDetectionFields);

  CreateGeneralOptions();

  // Apply initial state: combo defaults to "None", hide both path rows.
  UpdateLoopDetectionFields();
}

void SequentialMatchingTab::UpdateLoopDetectionFields() {
  QGridLayout* grid = options_widget_->findChild<QGridLayout*>();
  if (!grid) return;

  const int idx = loop_detection_type_cb_->currentIndex();

  // Helper: hide or show every widget in the two grid rows occupied by
  // one AddOptionFilePath call (label at row, lineEdit at row, button at
  // row+1).
  auto SetRowVisible = [grid](int row, bool visible) {
    for (int r = row; r <= row + 1; ++r) {
      for (int c = 0; c < 2; ++c) {
        QLayoutItem* item = grid->itemAtPosition(r, c);
        if (item && item->widget()) {
          item->widget()->setVisible(visible);
        }
      }
    }
  };

  // 0 = None, 1 = Vocab Tree, 2+ = global descriptor (MixVPR / MegaLoc / ...)
  SetRowVisible(vocab_tree_grid_row_, idx == 1);
#ifdef COLMAP_ONNX_ENABLED
  SetRowVisible(global_descriptor_grid_row_, idx >= 2);
#endif
}

void SequentialMatchingTab::Run() {
  WriteOptions();

  // Sync loop detection state from combo box.
  const int idx = loop_detection_type_cb_->currentIndex();
  options_->sequential_pairing->loop_detection = (idx != 0);
  if (idx == 1) {
    // Vocab Tree: clear global descriptor paths.
    options_->sequential_pairing->loop_detection_model_path.clear();
  }
#ifdef COLMAP_ONNX_ENABLED
  else if (idx >= 2) {
    // Global descriptor: clear vocab tree path, set model type from combo.
    options_->sequential_pairing->vocab_tree_path.clear();
    options_->sequential_pairing->loop_detection_model_type =
        loop_detection_type_cb_->currentText().toStdString();
    options_->sequential_pairing->loop_detection_database_path =
        *options_->database_path;
  }
#endif

  if (options_->sequential_pairing->loop_detection) {
#ifdef COLMAP_ONNX_ENABLED
    if (idx >= 2) {
      // Auto-derive image_path from the project.
      options_->sequential_pairing->loop_detection_image_path =
          *options_->image_path;
      if (options_->sequential_pairing->loop_detection_image_path.empty() ||
          !ExistsDir(
              options_->sequential_pairing->loop_detection_image_path)) {
        QMessageBox::critical(
            this, "",
            tr("Image path is not set or does not exist.\n\n"
               "For %1 loop detection, the image folder must be "
               "configured in the project settings (File > New Project).\n\n"
               "Current path: %2")
                .arg(loop_detection_type_cb_->currentText(),
                     QString::fromStdString(
                         options_->sequential_pairing
                             ->loop_detection_image_path.string())));
        return;
      }
      const auto& model_path =
          options_->sequential_pairing->loop_detection_model_path;
      if (!model_path.empty() && !ExistsFile(model_path) &&
          !IsURI(model_path.string())) {
        QMessageBox::critical(
            this, "",
            tr("Invalid %1 model path. Leave empty for "
               "auto-download, or provide a valid local file.")
                .arg(loop_detection_type_cb_->currentText()));
        return;
      }
    } else
#endif
    {
      const auto& tree_path =
          options_->sequential_pairing->vocab_tree_path;
      if (!tree_path.empty() && !ExistsFile(tree_path) &&
          !IsURI(tree_path.string())) {
        QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
        return;
      }
    }
  }

  auto matcher = CreateSequentialFeatureMatcher(*options_->sequential_pairing,
                                                *options_->feature_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

VocabTreeMatchingTab::VocabTreeMatchingTab(QWidget* parent,
                                           OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options_->vocab_tree_pairing->num_images,
                                "num_images");
  options_widget_->AddOptionInt(
      &options_->vocab_tree_pairing->num_nearest_neighbors,
      "num_nearest_neighbors");
  options_widget_->AddOptionInt(
      &options_->vocab_tree_pairing->num_checks, "num_checks", 1);
  options_widget_->AddOptionInt(
      &options_->vocab_tree_pairing->num_images_after_verification,
      "num_images_after_verification",
      0);
  options_widget_->AddOptionInt(
      &options_->vocab_tree_pairing->max_num_features, "max_num_features", -1);
  options_widget_->AddOptionFilePath(
      &options_->vocab_tree_pairing->vocab_tree_path, "vocab_tree_path");

  CreateGeneralOptions();
}

void VocabTreeMatchingTab::Run() {
  WriteOptions();

  // An empty path is valid: the matcher then resolves the default vocabulary
  // tree for the database's feature type via GetVocabTreeUriForFeatureType.
  if (!options_->vocab_tree_pairing->vocab_tree_path.empty() &&
      !ExistsFile(options_->vocab_tree_pairing->vocab_tree_path) &&
      !IsURI(options_->vocab_tree_pairing->vocab_tree_path.string())) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  auto matcher = CreateVocabTreeFeatureMatcher(*options_->vocab_tree_pairing,
                                               *options_->feature_matching,
                                               *options_->two_view_geometry,
                                               *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

#ifdef COLMAP_ONNX_ENABLED
GlobalDescriptorMatchingTab::GlobalDescriptorMatchingTab(
    QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  // Model type selector — populated from GlobalDescriptorModel registry.
  model_type_cb_ = new QComboBox(this);
  for (auto name : retrieval::GlobalDescriptorModel::ModelNames()) {
    model_type_cb_->addItem(QString::fromStdString(std::string(name)));
  }
  options_widget_->AddWidgetRow("Type", model_type_cb_);

  options_widget_->AddOptionInt(
      &options_->global_descriptor_pairing->num_images, "num_images");
  options_widget_->AddOptionInt(
      &options_->global_descriptor_pairing->batch_size, "batch_size", 1);
  model_path_edit_ = options_widget_->AddOptionFilePath(
      &options_->global_descriptor_pairing->model_path,
      "Model path<br>(ONNX, optional)");

  connect(model_type_cb_,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          this,
          &GlobalDescriptorMatchingTab::OnModelTypeChanged);

  // If model_path is empty, populate with the default for the current type.
  if (options_->global_descriptor_pairing->model_path.empty()) {
    OnModelTypeChanged(model_type_cb_->currentIndex());
  }

  CreateGeneralOptions();
}

void GlobalDescriptorMatchingTab::OnModelTypeChanged(int index) {
  // Set model_type and clear model_path (auto-download from registry).
  const QString name = model_type_cb_->itemText(index);
  options_->global_descriptor_pairing->model_type = name.toStdString();
  options_->global_descriptor_pairing->model_path.clear();

  if (model_path_edit_) {
    model_path_edit_->setText(QString());
  }
}

void GlobalDescriptorMatchingTab::Run() {
  WriteOptions();

  // Auto-derive paths from the project.
  options_->global_descriptor_pairing->image_path = *options_->image_path;
  options_->global_descriptor_pairing->database_path =
      *options_->database_path;

  const auto& model_path =
      options_->global_descriptor_pairing->model_path;
  // Empty path is valid: auto-download from HuggingFace.
  if (!model_path.empty() && !ExistsFile(model_path) &&
      !IsURI(model_path.string())) {
    QMessageBox::critical(
        this, "",
        tr("Invalid global descriptor model path. Leave empty for "
           "auto-download, or provide a valid local file or URL."));
    return;
  }

  if (options_->global_descriptor_pairing->image_path.empty()) {
    QMessageBox::critical(
        this, "",
        tr("Image path not set. Please set it in the project options."));
    return;
  }

  auto matcher = CreateGlobalDescriptorFeatureMatcher(
      *options_->global_descriptor_pairing,
      *options_->feature_matching,
      *options_->two_view_geometry,
      *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}
#endif  // COLMAP_ONNX_ENABLED

SpatialMatchingTab::SpatialMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionBool(&options_->spatial_pairing->ignore_z,
                                 "ignore_z");
  options_widget_->AddOptionInt(&options_->spatial_pairing->max_num_neighbors,
                                "max_num_neighbors");
  options_widget_->AddOptionInt(&options_->spatial_pairing->min_num_neighbors,
                                "min_num_neighbors");
  options_widget_->AddOptionDouble(&options_->spatial_pairing->max_distance,
                                   "max_distance");

  CreateGeneralOptions();
}

void SpatialMatchingTab::Run() {
  WriteOptions();

  auto matcher = CreateSpatialFeatureMatcher(*options_->spatial_pairing,
                                             *options_->feature_matching,
                                             *options_->two_view_geometry,
                                             *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

TransitiveMatchingTab::TransitiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options->transitive_pairing->batch_size,
                                "batch_size");
  options_widget_->AddOptionInt(&options->transitive_pairing->num_iterations,
                                "num_iterations");

  CreateGeneralOptions();
}

void TransitiveMatchingTab::Run() {
  WriteOptions();

  auto matcher = CreateTransitiveFeatureMatcher(*options_->transitive_pairing,
                                                *options_->feature_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

CustomMatchingTab::CustomMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  custom_match_type_cb_ = new QComboBox(this);
  custom_match_type_cb_->addItem(QString("Image pairs"));
  custom_match_type_cb_->addItem(QString("Raw feature matches"));
  custom_match_type_cb_->addItem(QString("Inlier feature matches"));
  options_widget_->AddOptionRow("type", custom_match_type_cb_, nullptr);

  options_widget_->AddOptionFilePath(&custom_match_list_path_,
                                     "match_list_path");
  options_widget_->AddOptionInt(
      &options_->imported_pairing->block_size, "block_size", 2);

  CreateGeneralOptions();
}

void CustomMatchingTab::Run() {
  WriteOptions();

  if (!ExistsFile(custom_match_list_path_)) {
    QMessageBox::critical(this, "", tr("Path does not exist!"));
    return;
  }

  std::unique_ptr<Thread> matcher;
  if (custom_match_type_cb_->currentIndex() == 0) {
    ImportedPairingOptions matcher_options;
    matcher_options.match_list_path = custom_match_list_path_;
    matcher = CreateImagePairsFeatureMatcher(matcher_options,
                                             *options_->feature_matching,
                                             *options_->two_view_geometry,
                                             *options_->database_path);
  } else {
    FeaturePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = custom_match_list_path_;
    if (custom_match_type_cb_->currentIndex() == 1) {
      matcher_options.verify_matches = true;
    } else if (custom_match_type_cb_->currentIndex() == 2) {
      matcher_options.verify_matches = false;
    }

    matcher = CreateFeaturePairsFeatureMatcher(matcher_options,
                                               *options_->feature_matching,
                                               *options_->two_view_geometry,
                                               *options_->database_path);
  }

  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

FeatureMatchingWidget::FeatureMatchingWidget(QWidget* parent,
                                             OptionManager* options)
    : parent_(parent) {
  // Do not change flag, to make sure feature database is not accessed from
  // multiple threads
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Feature matching");

  QGridLayout* grid = new QGridLayout(this);

  tab_widget_ = new QTabWidget(this);
  tab_widget_->addTab(new ExhaustiveMatchingTab(this, options),
                      tr("Exhaustive"));
  tab_widget_->addTab(new SequentialMatchingTab(this, options),
                      tr("Sequential"));
  tab_widget_->addTab(new VocabTreeMatchingTab(this, options), tr("VocabTree"));
#ifdef COLMAP_ONNX_ENABLED
  tab_widget_->addTab(new GlobalDescriptorMatchingTab(this, options),
                      tr("GlobalDescriptor"));
#endif
  tab_widget_->addTab(new SpatialMatchingTab(this, options), tr("Spatial"));
  tab_widget_->addTab(new TransitiveMatchingTab(this, options),
                      tr("Transitive"));
  tab_widget_->addTab(new CustomMatchingTab(this, options), tr("Custom"));

  grid->addWidget(tab_widget_, 0, 0);
}

void FeatureMatchingWidget::showEvent(QShowEvent* event) {
  parent_->setDisabled(true);
}

void FeatureMatchingWidget::hideEvent(QHideEvent* event) {
  parent_->setEnabled(true);
}

}  // namespace colmap
