#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_start_w_contribute(
    double* PinholeExtraCalib_precond_diag,
    unsigned int PinholeExtraCalib_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeExtraCalib_p,
    unsigned int PinholeExtraCalib_p_num_alloc,
    double* out_PinholeExtraCalib_w,
    unsigned int out_PinholeExtraCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar