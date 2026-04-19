#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_start_w_contribute(
    double* PinholePrincipalPoint_precond_diag,
    unsigned int PinholePrincipalPoint_precond_diag_num_alloc,
    const double* const diag,
    double* PinholePrincipalPoint_p,
    unsigned int PinholePrincipalPoint_p_num_alloc,
    double* out_PinholePrincipalPoint_w,
    unsigned int out_PinholePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar