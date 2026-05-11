#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndDistortionStartW(
    double *SimpleRadialFocalAndDistortion_precond_diag,
    unsigned int SimpleRadialFocalAndDistortion_precond_diag_num_alloc,
    const double *const diag, double *SimpleRadialFocalAndDistortion_p,
    unsigned int SimpleRadialFocalAndDistortion_p_num_alloc,
    double *out_SimpleRadialFocalAndDistortion_w,
    unsigned int out_SimpleRadialFocalAndDistortion_w_num_alloc,
    size_t problem_size);

} // namespace caspar