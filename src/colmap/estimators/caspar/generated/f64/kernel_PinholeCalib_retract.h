#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_retract(double* PinholeCalib,
                          unsigned int PinholeCalib_num_alloc,
                          double* delta,
                          unsigned int delta_num_alloc,
                          double* out_PinholeCalib_retracted,
                          unsigned int out_PinholeCalib_retracted_num_alloc,
                          size_t problem_size);

}  // namespace caspar