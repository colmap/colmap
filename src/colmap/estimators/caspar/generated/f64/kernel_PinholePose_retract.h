#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_retract(double* PinholePose,
                         unsigned int PinholePose_num_alloc,
                         double* delta,
                         unsigned int delta_num_alloc,
                         double* out_PinholePose_retracted,
                         unsigned int out_PinholePose_retracted_num_alloc,
                         size_t problem_size);

}  // namespace caspar