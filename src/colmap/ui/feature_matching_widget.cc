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
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"
#include "colmap/util/file.h"

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
};

class VocabTreeMatchingTab : public FeatureMatchingTab {
 public:
  VocabTreeMatchingTab(QWidget* parent, OptionManager* options);
  void Run() override;
};

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
  std::string custom_match_list_path_;
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
  add_matcher_type(FeatureMatcherType::SIFT);
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
  options_widget_->AddOptionInt(&options_->two_view_geometry->min_num_inliers,
                                "min_num_inliers");
  options_widget_->AddOptionBool(&options_->two_view_geometry->multiple_models,
                                 "multiple_models");
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
  options_widget_->AddOptionBool(&options_->sequential_pairing->loop_detection,
                                 "loop_detection");
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
  options_widget_->AddOptionFilePath(
      &options_->sequential_pairing->vocab_tree_path, "vocab_tree_path");

  CreateGeneralOptions();
}

void SequentialMatchingTab::Run() {
  WriteOptions();

  if (options_->sequential_pairing->loop_detection &&
      !ExistsFile(options_->sequential_pairing->vocab_tree_path) &&
      !IsURI(options_->sequential_pairing->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
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

  if (!ExistsFile(options_->vocab_tree_pairing->vocab_tree_path) &&
      !IsURI(options_->vocab_tree_pairing->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  auto matcher = CreateVocabTreeFeatureMatcher(*options_->vocab_tree_pairing,
                                               *options_->feature_matching,
                                               *options_->two_view_geometry,
                                               *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

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
