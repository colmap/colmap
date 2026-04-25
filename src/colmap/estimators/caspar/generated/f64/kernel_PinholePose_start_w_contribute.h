#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_start_w_contribute(
    double* PinholePose_precond_diag,
    unsigned int PinholePose_precond_diag_num_alloc,
    const double* const diag,
    double* PinholePose_p,
    unsigned int PinholePose_p_num_alloc,
    double* out_PinholePose_w,
    unsigned int out_PinholePose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar