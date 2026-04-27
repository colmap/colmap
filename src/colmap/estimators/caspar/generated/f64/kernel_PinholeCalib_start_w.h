#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_start_w(double* PinholeCalib_precond_diag,
                          unsigned int PinholeCalib_precond_diag_num_alloc,
                          const double* const diag,
                          double* PinholeCalib_p,
                          unsigned int PinholeCalib_p_num_alloc,
                          double* out_PinholeCalib_w,
                          unsigned int out_PinholeCalib_w_num_alloc,
                          size_t problem_size);

}  // namespace caspar