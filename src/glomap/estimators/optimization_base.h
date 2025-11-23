
#pragma once

#include <thread>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace glomap {

struct OptimizationBaseOptions {
  // The threshold for the loss function
  double thres_loss_function = 1e-1;

  // The options for the solver
  ceres::Solver::Options solver_options;

  OptimizationBaseOptions() {
    solver_options.num_threads = std::thread::hardware_concurrency();
    solver_options.max_num_iterations = 100;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.function_tolerance = 1e-5;
  }
};

}  // namespace glomap
