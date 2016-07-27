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

#include "ui/feature_matching_widget.h"

#include "base/feature_matching.h"
#include "ui/options_widget.h"

namespace colmap {

class FeatureMatchingTab : public OptionsWidget {
 public:
  FeatureMatchingTab() {}

  FeatureMatchingTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent), options_(options) {}

  virtual void Run() = 0;

 protected:
  void CreateProgressBar() {
    progress_bar_ = new QProgressDialog(this);
    progress_bar_->setWindowModality(Qt::ApplicationModal);
    progress_bar_->setLabel(new QLabel(tr("Matching..."), this));
    progress_bar_->setMaximum(0);
    progress_bar_->setMinimum(0);
    progress_bar_->setValue(0);
    progress_bar_->hide();
    progress_bar_->close();
  }

  void ShowProgressBar() {
    progress_bar_->show();
    progress_bar_->raise();
  }

  void CreateGeneralOptions() {
    AddSpacer();
    AddSpacer();
    AddSection("General Options");
    AddSpacer();

    AddOptionInt(&options_->match_options->num_threads, "num_threads", -1);
    AddOptionInt(&options_->match_options->gpu_index, "gpu_index", -1);
    AddOptionDouble(&options_->match_options->max_ratio, "max_ratio");
    AddOptionDouble(&options_->match_options->max_distance, "max_distance");
    AddOptionBool(&options_->match_options->cross_check, "cross_check");
    AddOptionInt(&options_->match_options->max_num_matches, "max_num_matches");
    AddOptionDouble(&options_->match_options->max_error, "max_error");
    AddOptionDouble(&options_->match_options->confidence, "confidence", 0, 1,
                    0.00001, 5);
    AddOptionInt(&options_->match_options->max_num_trials, "max_num_trials");
    AddOptionDouble(&options_->match_options->min_inlier_ratio,
                    "min_inlier_ratio", 0, 1, 0.001, 3);
    AddOptionInt(&options_->match_options->min_num_inliers, "min_num_inliers");
    AddOptionBool(&options_->match_options->multiple_models, "multiple_models");
    AddOptionBool(&options_->match_options->guided_matching, "guided_matching");

    AddSpacer();

    QPushButton* run_button = new QPushButton(tr("Run"), this);
    grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
    connect(run_button, &QPushButton::released, this, &FeatureMatchingTab::Run);
  }

  OptionManager* options_;
  QProgressDialog* progress_bar_;
};

class ExhaustiveMatchingTab : public FeatureMatchingTab {
 public:
  ExhaustiveMatchingTab(QWidget* parent, OptionManager* options)
      : FeatureMatchingTab(parent, options) {
    AddOptionInt(&options->exhaustive_match_options->block_size, "block_size");
    AddOptionBool(&options->exhaustive_match_options->preemptive, "preemptive");
    AddOptionInt(&options->exhaustive_match_options->preemptive_num_features,
                 "preemptive_num_features");
    AddOptionInt(&options->exhaustive_match_options->preemptive_min_num_matches,
                 "preemptive_min_num_matches");

    CreateGeneralOptions();
  }

  void Run() override {
    WriteOptions();

    ExhaustiveFeatureMatcher* feature_matcher = new ExhaustiveFeatureMatcher(
        options_->match_options->Options(),
        options_->exhaustive_match_options->Options(),
        *options_->database_path);

    feature_matcher->start();
    CreateProgressBar();

    connect(feature_matcher, &QThread::finished, progress_bar_,
            [this, feature_matcher]() {
              this->progress_bar_->hide();
              feature_matcher->deleteLater();
            });

    connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
      feature_matcher->Stop();
      feature_matcher->wait();
    });

    ShowProgressBar();
  }
};

class SequentialMatchingTab : public FeatureMatchingTab {
 public:
  SequentialMatchingTab(QWidget* parent, OptionManager* options)
      : FeatureMatchingTab(parent, options) {
    AddOptionInt(&options->sequential_match_options->overlap, "overlap");
    AddOptionBool(&options->sequential_match_options->loop_detection,
                  "loop_detection");
    AddOptionInt(&options->sequential_match_options->loop_detection_period,
                 "loop_detection_period");
    AddOptionInt(&options->sequential_match_options->loop_detection_num_images,
                 "loop_detection_num_images");
    AddOptionFilePath(&options->sequential_match_options->vocab_tree_path,
                      "vocab_tree_path");

    CreateGeneralOptions();
  }

  void Run() override {
    WriteOptions();

    if (options_->sequential_match_options->loop_detection &&
        !boost::filesystem::is_regular_file(
            options_->sequential_match_options->vocab_tree_path)) {
      QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
      return;
    }

    SequentialFeatureMatcher* feature_matcher = new SequentialFeatureMatcher(
        options_->match_options->Options(),
        options_->sequential_match_options->Options(),
        *options_->database_path);

    feature_matcher->start();
    CreateProgressBar();

    connect(feature_matcher, &QThread::finished, progress_bar_,
            [this, feature_matcher]() {
              this->progress_bar_->hide();
              feature_matcher->deleteLater();
            });

    connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
      feature_matcher->Stop();
      feature_matcher->wait();
    });

    ShowProgressBar();
  }
};

class VocabTreeMatchingTab : public FeatureMatchingTab {
 public:
  VocabTreeMatchingTab(QWidget* parent, OptionManager* options)
      : FeatureMatchingTab(parent, options) {
    AddOptionInt(&options->vocab_tree_match_options->num_images, "num_images");
    AddOptionFilePath(&options->vocab_tree_match_options->vocab_tree_path,
                      "vocab_tree_path");

    CreateGeneralOptions();
  }

  void Run() override {
    WriteOptions();

    if (!boost::filesystem::is_regular_file(
            options_->vocab_tree_match_options->vocab_tree_path)) {
      QMessageBox::critical(this, "", tr("Invalid vocabulary tree path."));
      return;
    }

    VocabTreeFeatureMatcher* feature_matcher = new VocabTreeFeatureMatcher(
        options_->match_options->Options(),
        options_->vocab_tree_match_options->Options(),
        *options_->database_path);

    feature_matcher->start();
    CreateProgressBar();

    connect(feature_matcher, &QThread::finished, progress_bar_,
            [this, feature_matcher]() {
              this->progress_bar_->hide();
              feature_matcher->deleteLater();
            });

    connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
      feature_matcher->Stop();
      feature_matcher->wait();
    });

    ShowProgressBar();
  }
};

class SpatialMatchingTab : public FeatureMatchingTab {
 public:
  SpatialMatchingTab(QWidget* parent, OptionManager* options)
      : FeatureMatchingTab(parent, options) {
    AddOptionBool(&options->spatial_match_options->is_gps, "is_gps");
    AddOptionBool(&options->spatial_match_options->ignore_z, "ignore_z");
    AddOptionInt(&options->spatial_match_options->max_num_neighbors,
                 "max_num_neighbors");
    AddOptionDouble(&options->spatial_match_options->max_distance,
                    "max_distance");

    CreateGeneralOptions();
  }

  void Run() override {
    WriteOptions();

    SpatialFeatureMatcher* feature_matcher = new SpatialFeatureMatcher(
        options_->match_options->Options(),
        options_->spatial_match_options->Options(), *options_->database_path);

    feature_matcher->start();
    CreateProgressBar();

    connect(feature_matcher, &QThread::finished, progress_bar_,
            [this, feature_matcher]() {
              this->progress_bar_->hide();
              feature_matcher->deleteLater();
            });

    connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
      feature_matcher->Stop();
      feature_matcher->wait();
    });

    ShowProgressBar();
  }
};

class CustomMatchingTab : public FeatureMatchingTab {
 public:
  CustomMatchingTab(QWidget* parent, OptionManager* options)
      : FeatureMatchingTab(parent, options) {
    match_type_cb_ = new QComboBox(this);
    match_type_cb_->addItem(QString("Image pairs"));
    match_type_cb_->addItem(QString("Raw feature matches"));
    match_type_cb_->addItem(QString("Inlier feature matches"));
    grid_layout_->addWidget(match_type_cb_, grid_layout_->rowCount(), 1);

    AddOptionFilePath(&match_list_path_, "match_list_path");

    CreateGeneralOptions();
  }

  void Run() override {
    WriteOptions();

    if (!boost::filesystem::exists(match_list_path_)) {
      QMessageBox::critical(this, "", tr("Path does not exist!"));
      return;
    }

    WriteOptions();

    // Match list selected
    if (match_type_cb_->currentIndex() == 0) {
      ImagePairsFeatureMatcher* feature_matcher = new ImagePairsFeatureMatcher(
          options_->match_options->Options(), *options_->database_path,
          match_list_path_);

      feature_matcher->start();
      CreateProgressBar();

      connect(feature_matcher, &QThread::finished, progress_bar_,
              [this]() { this->progress_bar_->hide(); });

      connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
        feature_matcher->Stop();
        feature_matcher->wait();
      });
    } else {
      bool compute_inliers = false;

      if (match_type_cb_->currentIndex() == 1) {
        compute_inliers = true;
      } else if (match_type_cb_->currentIndex() == 2) {
        compute_inliers = false;
      }

      FeaturePairsFeatureMatcher* feature_matcher =
          new FeaturePairsFeatureMatcher(
              options_->match_options->Options(), compute_inliers,
              *options_->database_path, match_list_path_);

      feature_matcher->start();
      CreateProgressBar();

      connect(feature_matcher, &QThread::finished, progress_bar_,
              [this]() { this->progress_bar_->hide(); });

      connect(progress_bar_, &QProgressDialog::canceled, [feature_matcher]() {
        feature_matcher->Stop();
        feature_matcher->wait();
      });
    }

    ShowProgressBar();
  }

 private:
  std::string match_list_path_;
  QComboBox* match_type_cb_;
};

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
