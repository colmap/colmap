/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <cuda_runtime.h>

namespace caspar {

size_t sort_indices_get_tmp_nbytes(size_t problem_size);

void sort_indices(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in,
                  uint* sorted_out, uint* target_out, uint* argsort_out, size_t problem_size);

size_t sort_keys_get_tmp_nbytes(size_t problem_size);

void sort_keys(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in, uint* sorted_out,
               size_t problem_size);

void select_index(const uint* const input, const uint* const selections, uint* const output,
                  size_t problem_size);

};  // namespace caspar
