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

#ifndef COLMAP_SRC_OPTIM_LEAST_ABSOLUTE_DEVIATIONS_H_
#define COLMAP_SRC_OPTIM_LEAST_ABSOLUTE_DEVIATIONS_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "util/logging.h"

namespace colmap {

struct LeastAbsoluteDeviationsOptions {
  // Augmented Lagrangian parameter.
  double rho = 1.0;

  // Over-relaxation parameter, typical values are between 1.0 and 1.8.
  double alpha = 1.0;

  // Maximum solver iterations.
  int max_num_iterations = 1000;

  // Absolute and relative solution thresholds, as suggested by Boyd et al.
  double absolute_tolerance = 1e-4;
  double relative_tolerance = 1e-2;
};

// Least absolute deviations (LAD) fitting via ADMM by solving the problem:
//
//        min || A x - b ||_1
//
// The solution is returned in the vector x and the iterative solver is
// initialized with the given value. This implementation is based on the paper
// "Distributed Optimization and Statistical Learning via the Alternating
// Direction Method of Multipliers" by Boyd et al. and the Matlab implementation
// at https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html
bool SolveLeastAbsoluteDeviations(const LeastAbsoluteDeviationsOptions& options,
                                  const Eigen::SparseMatrix<double>& A,
                                  const Eigen::VectorXd& b, Eigen::VectorXd* x);

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_LEAST_ABSOLUTE_DEVIATIONS_H_
