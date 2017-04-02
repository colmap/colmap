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

#ifndef COLMAP_SRC_OPTIM_SPRT_H_
#define COLMAP_SRC_OPTIM_SPRT_H_

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

  bool Evaluate(const std::vector<double>& residuals, const double max_residual,
                size_t* num_inliers, size_t* num_eval_samples);

 private:
  void UpdateDecisionThreshold();

  Options options_;
  double delta_epsilon_;
  double delta_1_epsilon_1_;
  double decision_threshold_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_SPRT_H_
