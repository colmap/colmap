// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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
