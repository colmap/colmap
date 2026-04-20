#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_update_step(
    double* SimpleRadialExtraCalib_step_k,
    unsigned int SimpleRadialExtraCalib_step_k_num_alloc,
    double* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialExtraCalib_step_kp1,
    unsigned int out_SimpleRadialExtraCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar