#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_pred_decrease_times_two(double* Point_step,
                                   unsigned int Point_step_num_alloc,
                                   double* Point_precond_diag,
                                   unsigned int Point_precond_diag_num_alloc,
                                   const double* const diag,
                                   double* Point_njtr,
                                   unsigned int Point_njtr_num_alloc,
                                   double* const out_Point_pred_dec,
                                   size_t problem_size);

}  // namespace caspar