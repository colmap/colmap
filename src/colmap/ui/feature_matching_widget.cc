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

#include "colmap/ui/feature_matching_widget.h"

#include "colmap/controllers/feature_matching.h"
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"

namespace colmap {

class FeatureMatchingTab : public QWidget {
 public:
  FeatureMatchingTab(QWidget* parent, OptionManager* options);

  virtual void Run() = 0;

 protected:
  void CreateGeneralOptions();

  OptionManager* options_;
  OptionsWidget* options_widget_;
  QGridLayout* grid_layout_;
  ThreadControlWidget* thread_control_widget_;
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
  std::string match_list_path_;
  QComboBox* match_type_cb_;
};

FeatureMatchingTab::FeatureMatchingTab(QWidget* parent, OptionManager* options)
    : QWidget(parent),
      options_(options),
      options_widget_(new OptionsWidget(this)),
      grid_layout_(new QGridLayout(this)),
      thread_control_widget_(new ThreadControlWidget(this)) {}

void FeatureMatchingTab::CreateGeneralOptions() {
  options_widget_->AddSpacer();
  options_widget_->AddSpacer();
  options_widget_->AddSection("General Options");
  options_widget_->AddSpacer();

  options_widget_->AddOptionInt(
      &options_->sift_matching->num_threads, "num_threads", -1);
  options_widget_->AddOptionBool(&options_->sift_matching->use_gpu, "use_gpu");
  options_widget_->AddOptionText(&options_->sift_matching->gpu_index,
                                 "gpu_index");
  options_widget_->AddOptionDouble(&options_->sift_matching->max_ratio,
                                   "max_ratio");
  options_widget_->AddOptionDouble(&options_->sift_matching->max_distance,
                                   "max_distance");
  options_widget_->AddOptionBool(&options_->sift_matching->cross_check,
                                 "cross_check");
  options_widget_->AddOptionInt(&options_->sift_matching->max_num_matches,
                                "max_num_matches");
  options_widget_->AddOptionBool(&options_->sift_matching->guided_matching,
                                 "guided_matching");
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

ExhaustiveMatchingTab::ExhaustiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(
      &options_->exhaustive_matching->block_size, "block_size", 2);

  CreateGeneralOptions();
}

void ExhaustiveMatchingTab::Run() {
  options_widget_->WriteOptions();

  auto matcher = CreateExhaustiveFeatureMatcher(*options_->exhaustive_matching,
                                                *options_->sift_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

SequentialMatchingTab::SequentialMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options_->sequential_matching->overlap,
                                "overlap");
  options_widget_->AddOptionBool(
      &options_->sequential_matching->quadratic_overlap, "quadratic_overlap");
  options_widget_->AddOptionBool(&options_->sequential_matching->loop_detection,
                                 "loop_detection");
  options_widget_->AddOptionInt(
      &options_->sequential_matching->loop_detection_period,
      "loop_detection_period");
  options_widget_->AddOptionInt(
      &options_->sequential_matching->loop_detection_num_images,
      "loop_detection_num_images");
  options_widget_->AddOptionInt(
      &options_->sequential_matching->loop_detection_num_nearest_neighbors,
      "loop_detection_num_nearest_neighbors");
  options_widget_->AddOptionInt(
      &options_->sequential_matching->loop_detection_num_checks,
      "loop_detection_num_checks",
      1);
  options_widget_->AddOptionInt(
      &options_->sequential_matching
           ->loop_detection_num_images_after_verification,
      "loop_detection_num_images_after_verification",
      0);
  options_widget_->AddOptionInt(
      &options_->sequential_matching->loop_detection_max_num_features,
      "loop_detection_max_num_features",
      -1);
  options_widget_->AddOptionFilePath(
      &options_->sequential_matching->vocab_tree_path, "vocab_tree_path");

  CreateGeneralOptions();
}

void SequentialMatchingTab::Run() {
  options_widget_->WriteOptions();

  if (options_->sequential_matching->loop_detection &&
      !ExistsFile(options_->sequential_matching->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  auto matcher = CreateSequentialFeatureMatcher(*options_->sequential_matching,
                                                *options_->sift_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

VocabTreeMatchingTab::VocabTreeMatchingTab(QWidget* parent,
                                           OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options_->vocab_tree_matching->num_images,
                                "num_images");
  options_widget_->AddOptionInt(
      &options_->vocab_tree_matching->num_nearest_neighbors,
      "num_nearest_neighbors");
  options_widget_->AddOptionInt(
      &options_->vocab_tree_matching->num_checks, "num_checks", 1);
  options_widget_->AddOptionInt(
      &options_->vocab_tree_matching->num_images_after_verification,
      "num_images_after_verification",
      0);
  options_widget_->AddOptionInt(
      &options_->vocab_tree_matching->max_num_features, "max_num_features", -1);
  options_widget_->AddOptionFilePath(
      &options_->vocab_tree_matching->vocab_tree_path, "vocab_tree_path");

  CreateGeneralOptions();
}

void VocabTreeMatchingTab::Run() {
  options_widget_->WriteOptions();

  if (!ExistsFile(options_->vocab_tree_matching->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  auto matcher = CreateVocabTreeFeatureMatcher(*options_->vocab_tree_matching,
                                               *options_->sift_matching,
                                               *options_->two_view_geometry,
                                               *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

SpatialMatchingTab::SpatialMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionBool(&options_->spatial_matching->ignore_z,
                                 "ignore_z");
  options_widget_->AddOptionInt(&options_->spatial_matching->max_num_neighbors,
                                "max_num_neighbors");
  options_widget_->AddOptionDouble(&options_->spatial_matching->max_distance,
                                   "max_distance");

  CreateGeneralOptions();
}

void SpatialMatchingTab::Run() {
  options_widget_->WriteOptions();

  auto matcher = CreateSpatialFeatureMatcher(*options_->spatial_matching,
                                             *options_->sift_matching,
                                             *options_->two_view_geometry,
                                             *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

TransitiveMatchingTab::TransitiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  options_widget_->AddOptionInt(&options->transitive_matching->batch_size,
                                "batch_size");
  options_widget_->AddOptionInt(&options->transitive_matching->num_iterations,
                                "num_iterations");

  CreateGeneralOptions();
}

void TransitiveMatchingTab::Run() {
  options_widget_->WriteOptions();

  auto matcher = CreateTransitiveFeatureMatcher(*options_->transitive_matching,
                                                *options_->sift_matching,
                                                *options_->two_view_geometry,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, std::move(matcher));
}

CustomMatchingTab::CustomMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  match_type_cb_ = new QComboBox(this);
  match_type_cb_->addItem(QString("Image pairs"));
  match_type_cb_->addItem(QString("Raw feature matches"));
  match_type_cb_->addItem(QString("Inlier feature matches"));
  options_widget_->AddOptionRow("type", match_type_cb_, nullptr);

  options_widget_->AddOptionFilePath(&match_list_path_, "match_list_path");
  options_widget_->AddOptionInt(
      &options_->image_pairs_matching->block_size, "block_size", 2);

  CreateGeneralOptions();
}

void CustomMatchingTab::Run() {
  options_widget_->WriteOptions();

  if (!ExistsFile(match_list_path_)) {
    QMessageBox::critical(this, "", tr("Path does not exist!"));
    return;
  }

  std::unique_ptr<Thread> matcher;
  if (match_type_cb_->currentIndex() == 0) {
    ImagePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path_;
    matcher = CreateImagePairsFeatureMatcher(matcher_options,
                                             *options_->sift_matching,
                                             *options_->two_view_geometry,
                                             *options_->database_path);
  } else {
    FeaturePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path_;
    if (match_type_cb_->currentIndex() == 1) {
      matcher_options.verify_matches = true;
    } else if (match_type_cb_->currentIndex() == 2) {
      matcher_options.verify_matches = false;
    }

    matcher = CreateFeaturePairsFeatureMatcher(matcher_options,
                                               *options_->sift_matching,
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
  setWindowFlags(Qt::Window);
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
