#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_step_first(
    double* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialCalib_step_kp1,
    unsigned int out_SimpleRadialCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar