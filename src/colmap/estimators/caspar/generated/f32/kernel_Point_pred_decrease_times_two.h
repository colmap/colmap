#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_pred_decrease_times_two(float* Point_step,
                                   unsigned int Point_step_num_alloc,
                                   float* Point_precond_diag,
                                   unsigned int Point_precond_diag_num_alloc,
                                   const float* const diag,
                                   float* Point_njtr,
                                   unsigned int Point_njtr_num_alloc,
                                   float* const out_Point_pred_dec,
                                   size_t problem_size);

}  // namespace caspar