// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "ui/feature_matching_widget.h"

#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"

namespace colmap {

class FeatureMatchingTab : public OptionsWidget {
 public:
  FeatureMatchingTab(QWidget* parent, OptionManager* options);

  virtual void Run() = 0;

 protected:
  void CreateGeneralOptions();

  OptionManager* options_;
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
    : OptionsWidget(parent),
      options_(options),
      thread_control_widget_(new ThreadControlWidget(this)) {}

void FeatureMatchingTab::CreateGeneralOptions() {
  AddSpacer();
  AddSpacer();
  AddSection("General Options");
  AddSpacer();

  AddOptionInt(&options_->sift_matching->num_threads, "num_threads", -1);
  AddOptionBool(&options_->sift_matching->use_gpu, "use_gpu");
  AddOptionText(&options_->sift_matching->gpu_index, "gpu_index");
  AddOptionDouble(&options_->sift_matching->max_ratio, "max_ratio");
  AddOptionDouble(&options_->sift_matching->max_distance, "max_distance");
  AddOptionBool(&options_->sift_matching->cross_check, "cross_check");
  AddOptionInt(&options_->sift_matching->max_num_matches, "max_num_matches");
  AddOptionDouble(&options_->sift_matching->max_error, "max_error");
  AddOptionDouble(&options_->sift_matching->confidence, "confidence", 0, 1,
                  0.00001, 5);
  AddOptionInt(&options_->sift_matching->max_num_trials, "max_num_trials");
  AddOptionDouble(&options_->sift_matching->min_inlier_ratio,
                  "min_inlier_ratio", 0, 1, 0.001, 3);
  AddOptionInt(&options_->sift_matching->min_num_inliers, "min_num_inliers");
  AddOptionBool(&options_->sift_matching->multiple_models, "multiple_models");
  AddOptionBool(&options_->sift_matching->guided_matching, "guided_matching");

  AddSpacer();

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
  connect(run_button, &QPushButton::released, this, &FeatureMatchingTab::Run);
}

ExhaustiveMatchingTab::ExhaustiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  AddOptionInt(&options_->exhaustive_matching->block_size, "block_size", 2);

  CreateGeneralOptions();
}

void ExhaustiveMatchingTab::Run() {
  WriteOptions();

  Thread* matcher = new ExhaustiveFeatureMatcher(*options_->exhaustive_matching,
                                                 *options_->sift_matching,
                                                 *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, matcher);
}

SequentialMatchingTab::SequentialMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  AddOptionInt(&options_->sequential_matching->overlap, "overlap");
  AddOptionBool(&options_->sequential_matching->loop_detection,
                "loop_detection");
  AddOptionInt(&options_->sequential_matching->loop_detection_period,
               "loop_detection_period");
  AddOptionInt(&options_->sequential_matching->loop_detection_num_images,
               "loop_detection_num_images");
  AddOptionInt(&options_->sequential_matching->loop_detection_num_verifications,
               "loop_detection_num_verifications");
  AddOptionInt(&options_->sequential_matching->loop_detection_max_num_features,
               "loop_detection_max_num_features", -1);
  AddOptionFilePath(&options_->sequential_matching->vocab_tree_path,
                    "vocab_tree_path");

  CreateGeneralOptions();
}

void SequentialMatchingTab::Run() {
  WriteOptions();

  if (options_->sequential_matching->loop_detection &&
      !ExistsFile(options_->sequential_matching->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  Thread* matcher = new SequentialFeatureMatcher(*options_->sequential_matching,
                                                 *options_->sift_matching,
                                                 *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, matcher);
}

VocabTreeMatchingTab::VocabTreeMatchingTab(QWidget* parent,
                                           OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  AddOptionInt(&options_->vocab_tree_matching->num_images, "num_images");
  AddOptionInt(&options_->vocab_tree_matching->num_verifications,
               "num_verifications");
  AddOptionInt(&options_->vocab_tree_matching->max_num_features,
               "max_num_features", -1);
  AddOptionFilePath(&options_->vocab_tree_matching->vocab_tree_path,
                    "vocab_tree_path");

  CreateGeneralOptions();
}

void VocabTreeMatchingTab::Run() {
  WriteOptions();

  if (!ExistsFile(options_->vocab_tree_matching->vocab_tree_path)) {
    QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
    return;
  }

  Thread* matcher = new VocabTreeFeatureMatcher(*options_->vocab_tree_matching,
                                                *options_->sift_matching,
                                                *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, matcher);
}

SpatialMatchingTab::SpatialMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  AddOptionBool(&options_->spatial_matching->is_gps, "is_gps");
  AddOptionBool(&options_->spatial_matching->ignore_z, "ignore_z");
  AddOptionInt(&options_->spatial_matching->max_num_neighbors,
               "max_num_neighbors");
  AddOptionDouble(&options_->spatial_matching->max_distance, "max_distance");

  CreateGeneralOptions();
}

void SpatialMatchingTab::Run() {
  WriteOptions();

  Thread* matcher = new SpatialFeatureMatcher(*options_->spatial_matching,
                                              *options_->sift_matching,
                                              *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, matcher);
}

TransitiveMatchingTab::TransitiveMatchingTab(QWidget* parent,
                                             OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  AddOptionInt(&options->transitive_matching->batch_size, "batch_size");
  AddOptionInt(&options->transitive_matching->num_iterations, "num_iterations");

  CreateGeneralOptions();
}

void TransitiveMatchingTab::Run() {
  WriteOptions();

  Thread* matcher = new TransitiveFeatureMatcher(*options_->transitive_matching,
                                                 *options_->sift_matching,
                                                 *options_->database_path);
  thread_control_widget_->StartThread("Matching...", true, matcher);
}

CustomMatchingTab::CustomMatchingTab(QWidget* parent, OptionManager* options)
    : FeatureMatchingTab(parent, options) {
  match_type_cb_ = new QComboBox(this);
  match_type_cb_->addItem(QString("Image pairs"));
  match_type_cb_->addItem(QString("Raw feature matches"));
  match_type_cb_->addItem(QString("Inlier feature matches"));
  grid_layout_->addWidget(match_type_cb_, grid_layout_->rowCount(), 1);

  AddOptionFilePath(&match_list_path_, "match_list_path");

  CreateGeneralOptions();
}

void CustomMatchingTab::Run() {
  WriteOptions();

  if (!ExistsFile(match_list_path_)) {
    QMessageBox::critical(this, "", tr("Path does not exist!"));
    return;
  }

  Thread* matcher = nullptr;
  if (match_type_cb_->currentIndex() == 0) {
    ImagePairsFeatureMatcher::Options matcher_options;
    matcher_options.match_list_path = match_list_path_;
    matcher = new ImagePairsFeatureMatcher(
        matcher_options, *options_->sift_matching, *options_->database_path);
  } else {
    FeaturePairsFeatureMatcher::Options matcher_options;
    matcher_options.match_list_path = match_list_path_;
    if (match_type_cb_->currentIndex() == 1) {
      matcher_options.verify_matches = true;
    } else if (match_type_cb_->currentIndex() == 2) {
      matcher_options.verify_matches = false;
    }

    matcher = new FeaturePairsFeatureMatcher(
        matcher_options, *options_->sift_matching, *options_->database_path);
  }

  thread_control_widget_->StartThread("Matching...", true, matcher);
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
