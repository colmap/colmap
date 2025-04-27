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

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace colmap {

// Sequential Probability Ratio Test as proposed in
//
//   "Randomized RANSAC with Sequential Probability Ratio Test",
//   Matas et al., 2005
class SPRT {
 public:
  struct Options {
    // Probability of rejecting a good model.
    double delta = 0.01;

    // A priori assumed minimum inlier ratio
    double epsilon = 0.1;

    // The ratio of the time it takes to estimate a model from a random sample
    // over the time it takes to decide whether one data sample is an
    // inlier or not. Matas et al. propose 200 for the 7-point algorithm.
    double eval_time_ratio = 200;

    // Number of models per random sample, that have to be verified. E.g. 1-3
    // for the 7-point fundamental matrix algorithm, or 1-10 for the 5-point
    // essential matrix algorithm.
    int num_models_per_sample = 1;
  };

  explicit SPRT(const Options& options);

  void Update(const Options& options);

  bool Evaluate(const std::vector<double>& residuals,
                double max_residual,
                size_t* num_inliers,
                size_t* num_eval_samples);

 private:
  void UpdateDecisionThreshold();

  Options options_;
  double delta_epsilon_;
  double delta_1_epsilon_1_;
  double decision_threshold_;
};

}  // namespace colmap
