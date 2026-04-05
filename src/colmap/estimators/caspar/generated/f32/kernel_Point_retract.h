#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_retract(float* Point,
                   unsigned int Point_num_alloc,
                   float* delta,
                   unsigned int delta_num_alloc,
                   float* out_Point_retracted,
                   unsigned int out_Point_retracted_num_alloc,
                   size_t problem_size);

}  // namespace caspar