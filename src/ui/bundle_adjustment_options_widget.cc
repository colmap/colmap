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

#include "ui/bundle_adjustment_options_widget.h"

namespace colmap {

BundleAdjustmentOptionsWidget::BundleAdjustmentOptionsWidget(
    QWidget* parent, OptionManager* options)
    : OptionsWidget(parent) {
  setWindowTitle("Bundle adjustment options");

  AddOptionInt(&options->ba_options->max_num_iterations, "max_num_iterations");
  AddOptionInt(&options->ba_options->max_linear_solver_iterations,
               "max_linear_solver_iterations");

  AddOptionDoubleLog(&options->ba_options->function_tolerance,
                     "function_tolerance [10eX]", -1000, 1000);
  AddOptionDoubleLog(&options->ba_options->gradient_tolerance,
                     "gradient_tolerance [10eX]", -1000, 1000);
  AddOptionDoubleLog(&options->ba_options->parameter_tolerance,
                     "parameter_tolerance [10eX]", -1000, 1000);

  AddOptionBool(&options->ba_options->refine_focal_length,
                "refine_focal_length");
  AddOptionBool(&options->ba_options->refine_principal_point,
                "refine_principal_point");
  AddOptionBool(&options->ba_options->refine_extra_params,
                "refine_extra_params");
}

}  // namespace colmap
