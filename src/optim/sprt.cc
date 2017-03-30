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

#include "optim/sprt.h"

namespace colmap {

SPRT::SPRT(const Options& options) { Update(options); }

void SPRT::Update(const Options& options) {
  options_ = options;
  delta_epsilon_ = options.delta / options.epsilon;
  delta_1_epsilon_1_ = (1 - options.delta) / (1 - options.epsilon);
  UpdateDecisionThreshold();
}

bool SPRT::Evaluate(const std::vector<double>& residuals,
                    const double max_residual, size_t* num_inliers,
                    size_t* num_eval_samples) {
  num_inliers = 0;

  double likelihood_ratio = 1;

  for (size_t i = 0; i < residuals.size(); ++i) {
    if (std::abs(residuals[i]) <= max_residual) {
      num_inliers += 1;
      likelihood_ratio *= delta_epsilon_;
    } else {
      likelihood_ratio *= delta_1_epsilon_1_;
    }

    if (likelihood_ratio > decision_threshold_) {
      *num_eval_samples = i + 1;
      return false;
    }
  }

  *num_eval_samples = residuals.size();

  return true;
}

void SPRT::UpdateDecisionThreshold() {
  // Equation 2
  const double C = (1 - options_.delta) *
                       std::log((1 - options_.delta) / (1 - options_.epsilon)) +
                   options_.delta * std::log(options_.delta / options_.epsilon);

  // Equation 6
  const double A0 =
      options_.eval_time_ratio * C / options_.num_models_per_sample + 1;

  double A = A0;

  const double kEps = 1.5e-8;

  // Compute A using the recursive relation
  //    A* = lim(n->inf) A
  // The series typically converges within 4 iterations

  for (size_t i = 0; i < 100; ++i) {
    const double A1 = A0 + std::log(A);

    if (std::abs(A1 - A) < kEps) {
      break;
    }

    A = A1;
  }

  decision_threshold_ = A;
}

}  // namespace colmap
