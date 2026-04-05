#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_start_w_contribute(
    double* SimpleRadialCalib_precond_diag,
    unsigned int SimpleRadialCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialCalib_p,
    unsigned int SimpleRadialCalib_p_num_alloc,
    double* out_SimpleRadialCalib_w,
    unsigned int out_SimpleRadialCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar