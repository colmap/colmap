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

#include "ui/reconstruction_options_widget.h"

namespace colmap {
namespace {

class GeneralTab : public OptionsWidget {
 public:
  GeneralTab(QWidget* parent, OptionManager* options) : OptionsWidget(parent) {
    IncrementalMapperOptions& inc_mapper_options =
        options->sparse_mapper_options->incremental_mapper;
    AddSection("Absolute Pose");
    AddOptionDouble(&inc_mapper_options.abs_pose_max_error,
                    "abs_pose_max_error [px]");
    AddOptionInt(&inc_mapper_options.abs_pose_min_num_inliers,
                 "abs_pose_min_num_inliers");
    AddOptionDouble(&inc_mapper_options.abs_pose_min_inlier_ratio,
                    "abs_pose_min_inlier_ratio");
    AddOptionInt(&inc_mapper_options.max_reg_trials, "max_reg_trials");

    AddSpacer();

    AddSection("Other");
    AddOptionBool(&options->sparse_mapper_options->extract_colors,
                  "extract_colors");
    AddOptionInt(&options->sparse_mapper_options->num_threads, "num_threads",
                 -1);
    AddOptionInt(&options->sparse_mapper_options->min_num_matches,
                 "min_num_matches");
    AddOptionBool(&options->sparse_mapper_options->ignore_watermarks,
                  "ignore_watermarks");
  };
};

class TriangulationTab : public OptionsWidget {
 public:
  TriangulationTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    TriangulationOptions& tri_options =
        options->sparse_mapper_options->triangulation;
    AddOptionInt(&tri_options.max_transitivity, "max_transitivity");
    AddOptionDouble(&tri_options.create_max_angle_error,
                    "create_max_angle_error [deg]");
    AddOptionDouble(&tri_options.continue_max_angle_error,
                    "continue_max_angle_error [deg]");
    AddOptionDouble(&tri_options.merge_max_reproj_error,
                    "merge_max_reproj_error [px]");
    AddOptionDouble(&tri_options.re_max_angle_error,
                    "re_max_angle_error [deg]");
    AddOptionDouble(&tri_options.re_min_ratio, "re_min_ratio");
    AddOptionInt(&tri_options.re_max_trials, "re_max_trials");
    AddOptionDouble(&tri_options.complete_max_reproj_error,
                    "complete_max_reproj_error [px]");
    AddOptionInt(&tri_options.complete_max_transitivity,
                 "complete_max_transitivity");
    AddOptionDouble(&tri_options.min_angle, "min_angle [deg]", 0, 180);
    AddOptionBool(&tri_options.ignore_two_view_tracks,
                  "ignore_two_view_tracks");
  };
};

class InitTab : public OptionsWidget {
 public:
  InitTab(QWidget* parent, OptionManager* options) : OptionsWidget(parent) {
    AddOptionInt(&options->sparse_mapper_options->init_image_id1,
                 "init_image_id1", -1);
    AddOptionInt(&options->sparse_mapper_options->init_image_id2,
                 "init_image_id2", -1);
    AddOptionInt(&options->sparse_mapper_options->init_num_trials,
                 "init_num_trials");

    IncrementalMapperOptions& inc_mapper_options =
        options->sparse_mapper_options->incremental_mapper;
    AddOptionInt(&inc_mapper_options.init_min_num_inliers,
                 "init_min_num_inliers");
    AddOptionDouble(&inc_mapper_options.init_max_error, "init_max_error");
    AddOptionDouble(&inc_mapper_options.init_max_forward_motion,
                    "init_max_forward_motion");
    AddOptionDouble(&inc_mapper_options.init_min_tri_angle,
                    "init_min_tri_angle [deg]");
  };
};

class MultiModelTab : public OptionsWidget {
 public:
  MultiModelTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionBool(&options->sparse_mapper_options->multiple_models,
                  "multiple_models");
    AddOptionInt(&options->sparse_mapper_options->max_num_models,
                 "max_num_models");
    AddOptionInt(&options->sparse_mapper_options->max_model_overlap,
                 "max_model_overlap");
    AddOptionInt(&options->sparse_mapper_options->min_model_size,
                 "min_model_size");
  };
};

class BundleAdjustmentTab : public OptionsWidget {
 public:
  BundleAdjustmentTab(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddSection("Camera parameters");
    AddOptionBool(&options->sparse_mapper_options->ba_refine_focal_length,
                  "refine_focal_length");
    AddOptionBool(&options->sparse_mapper_options->ba_refine_principal_point,
                  "refine_principal_point");
    AddOptionBool(&options->sparse_mapper_options->ba_refine_extra_params,
                  "refine_extra_params");

    AddSpacer();

    AddSection("Local Bundle Adjustment");
    AddOptionInt(&options->sparse_mapper_options->ba_local_num_images,
                 "num_images");
    AddOptionInt(&options->sparse_mapper_options->ba_local_max_num_iterations,
                 "max_num_iterations");
    AddOptionInt(&options->sparse_mapper_options->ba_local_max_refinements,
                 "max_refinements", 1);
    AddOptionDouble(
        &options->sparse_mapper_options->ba_local_max_refinement_change,
        "max_refinement_change", 0, 1, 1e-6, 6);

    AddSpacer();

    AddSection("Global Bundle Adjustment");
    AddOptionBool(&options->sparse_mapper_options->ba_global_use_pba,
                  "use_pba\n(requires SIMPLE_RADIAL)");
    AddOptionDouble(&options->sparse_mapper_options->ba_global_images_ratio,
                    "images_ratio");
    AddOptionInt(&options->sparse_mapper_options->ba_global_images_freq,
                 "images_freq");
    AddOptionDouble(&options->sparse_mapper_options->ba_global_points_ratio,
                    "points_ratio");
    AddOptionInt(&options->sparse_mapper_options->ba_global_points_freq,
                 "points_freq");
    AddOptionInt(&options->sparse_mapper_options->ba_global_max_num_iterations,
                 "max_num_iterations");
    AddOptionInt(&options->sparse_mapper_options->ba_global_pba_gpu_index,
                 "pba_gpu_index", -1);
    AddOptionInt(&options->sparse_mapper_options->ba_global_max_refinements,
                 "max_refinements", 1);
    AddOptionDouble(
        &options->sparse_mapper_options->ba_global_max_refinement_change,
        "max_refinement_change", 0, 1, 1e-6, 6);
  };
};

class FilterTab : public OptionsWidget {
 public:
  FilterTab(QWidget* parent, OptionManager* options) : OptionsWidget(parent) {
    AddOptionDouble(&options->sparse_mapper_options->min_focal_length_ratio,
                    "min_focal_length_ratio");
    AddOptionDouble(&options->sparse_mapper_options->max_focal_length_ratio,
                    "max_focal_length_ratio");
    AddOptionDouble(&options->sparse_mapper_options->max_extra_param,
                    "max_extra_param");

    IncrementalMapperOptions& inc_mapper_options =
        options->sparse_mapper_options->incremental_mapper;
    AddOptionDouble(&inc_mapper_options.filter_max_reproj_error,
                    "filter_max_reproj_error [px]");
    AddOptionDouble(&inc_mapper_options.filter_min_tri_angle,
                    "filter_min_tri_angle [deg]");
  };
};

}  // namespace

ReconstructionOptionsWidget::ReconstructionOptionsWidget(QWidget* parent,
                                                         OptionManager* options)
    : QWidget(parent) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Reconstruction options");

  QGridLayout* grid = new QGridLayout(this);

  QTabWidget* tab_widget = new QTabWidget(this);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget->addTab(new GeneralTab(this, options), tr("General"));
  tab_widget->addTab(new InitTab(this, options), tr("Init"));
  tab_widget->addTab(new TriangulationTab(this, options), tr("Triangulation"));
  tab_widget->addTab(new MultiModelTab(this, options), tr("Multi-Model"));
  tab_widget->addTab(new BundleAdjustmentTab(this, options), tr("Bundle-Adj."));
  tab_widget->addTab(new FilterTab(this, options), tr("Filter"));

  grid->addWidget(tab_widget, 0, 0);
}

}  // namespace colmap
