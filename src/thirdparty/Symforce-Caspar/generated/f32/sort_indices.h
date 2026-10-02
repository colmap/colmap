/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <cuda_runtime.h>

namespace caspar {

size_t SortIndicesGetTmpNbytes(size_t problem_size);

void SortIndices(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in, uint* sorted_out,
                 uint* target_out, uint* argsort_out, size_t problem_size);

size_t SortKeysGetTmpNbytes(size_t problem_size);

void SortKeys(void* tmp_storage, size_t tmp_storage_bytes, const uint* keys_in, uint* sorted_out,
              size_t problem_size);

void SelectIndex(const uint* const input, const uint* const selections, uint* const output,
                 size_t problem_size);

}  // namespace caspar
