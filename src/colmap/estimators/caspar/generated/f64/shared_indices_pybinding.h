/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "pybind_array_tools.h"
#include "shared_indices.h"

namespace caspar {

void shared_indices_pybinding(pybind11::object indices, pybind11::object indices_out) {
  int problem_size = GetNumRows(indices);
  AssertNumRowsEquals(indices_out, problem_size);
  const int n_blocks = (problem_size + 1024 - 1) / 1024;

  shared_indices(AsUintPtr(indices), reinterpret_cast<SharedIndex*>(AsUint2Ptr(indices_out)),
                 problem_size);
}

}  // namespace caspar
