#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_start_w(float* Point_precond_diag,
                   unsigned int Point_precond_diag_num_alloc,
                   const float* const diag,
                   float* Point_p,
                   unsigned int Point_p_num_alloc,
                   float* out_Point_w,
                   unsigned int out_Point_w_num_alloc,
                   size_t problem_size);

}  // namespace caspar