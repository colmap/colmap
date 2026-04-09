/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <pybind11/pybind11.h>

#include "solver_params.h"

namespace py = pybind11;

namespace caspar {

using SolverParamsT = SolverParams<double>;
void add_solver_params_pybinding(py::module_ module) {
  py::class_<SolverParamsT>(module, "SolverParams", "Class for setting solver parameters.")
      .def(py::init<>())
      .def_readwrite("solver_iter_max", &SolverParamsT::solver_iter_max)
      .def_readwrite("pcg_iter_max", &SolverParamsT::pcg_iter_max)
      .def_readwrite("diag_init", &SolverParamsT::diag_init)
      .def_readwrite("diag_scaling_up", &SolverParamsT::diag_scaling_up)
      .def_readwrite("diag_scaling_down", &SolverParamsT::diag_scaling_down)
      .def_readwrite("diag_exit_value", &SolverParamsT::diag_exit_value)
      .def_readwrite("diag_min", &SolverParamsT::diag_min)
      .def_readwrite("solver_rel_decrease_min", &SolverParamsT::solver_rel_decrease_min)
      .def_readwrite("score_exit_value", &SolverParamsT::score_exit_value)
      .def_readwrite("pcg_rel_decrease_min", &SolverParamsT::pcg_rel_decrease_min)
      .def_readwrite("pcg_rel_error_exit", &SolverParamsT::pcg_rel_error_exit)
      .def_readwrite("pcg_rel_score_exit", &SolverParamsT::pcg_rel_score_exit);
}

}  // namespace caspar
