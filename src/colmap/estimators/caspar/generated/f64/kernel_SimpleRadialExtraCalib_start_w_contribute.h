#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_start_w_contribute(
    double* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialExtraCalib_p,
    unsigned int SimpleRadialExtraCalib_p_num_alloc,
    double* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar