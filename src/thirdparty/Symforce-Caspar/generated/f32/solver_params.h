/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

template <typename T>
struct SolverParams {
  T diag_init = 1.0;
  T diag_scaling_up = 2.0;
  T diag_scaling_down = 0.333333f;
  T diag_exit_value = 1e3;
  T diag_min = 1e-12;
  T score_exit_value = 0.0f;
  int solver_iter_max = 100;
  int pcg_iter_max = 20;
  T pcg_rel_error_exit = 1e-4;
  T pcg_rel_score_exit = -1.0f;    // disabled if == -1.0f
  T pcg_rel_decrease_min = -1.0f;  // disabled if == -1.0f
  T solver_rel_decrease_min = 1.0f;

  // Templated conversion constructor to enable automatic type conversion
  SolverParams() = default;

  template <typename U>
  SolverParams(const SolverParams<U>& other)
      : diag_init(static_cast<T>(other.diag_init)),
        diag_scaling_up(static_cast<T>(other.diag_scaling_up)),
        diag_scaling_down(static_cast<T>(other.diag_scaling_down)),
        diag_exit_value(static_cast<T>(other.diag_exit_value)),
        diag_min(static_cast<T>(other.diag_min)),
        score_exit_value(static_cast<T>(other.score_exit_value)),
        solver_iter_max(other.solver_iter_max),
        pcg_iter_max(other.pcg_iter_max),
        pcg_rel_error_exit(static_cast<T>(other.pcg_rel_error_exit)),
        pcg_rel_score_exit(static_cast<T>(other.pcg_rel_score_exit)),
        pcg_rel_decrease_min(static_cast<T>(other.pcg_rel_decrease_min)),
        solver_rel_decrease_min(static_cast<T>(other.solver_rel_decrease_min)) {}
};
}  // namespace caspar
