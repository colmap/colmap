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
