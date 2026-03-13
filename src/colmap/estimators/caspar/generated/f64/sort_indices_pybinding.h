/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "pybind_array_tools.h"
#include "sort_indices.h"

namespace caspar {

void sort_indices_pybinding(pybind11::object keys_in, pybind11::object sorted_out,
                            pybind11::object target_out, pybind11::object argsort_out,
                            pybind11::object tmp_storage) {
  size_t problem_size = GetNumRows(keys_in);
  AssertNumRowsEquals(sorted_out, problem_size);
  AssertNumRowsEquals(target_out, problem_size);
  AssertNumRowsEquals(argsort_out, problem_size);
  size_t tmp_storage_bytes = GetNumRows(tmp_storage);

  sort_indices(AsCharPtr(tmp_storage), tmp_storage_bytes, AsUintPtr(keys_in), AsUintPtr(sorted_out),
               AsUintPtr(target_out), AsUintPtr(argsort_out), problem_size);
}

}  // namespace caspar
