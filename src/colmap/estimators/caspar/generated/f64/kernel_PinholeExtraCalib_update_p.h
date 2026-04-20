#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_p(
    double* PinholeExtraCalib_z,
    unsigned int PinholeExtraCalib_z_num_alloc,
    double* PinholeExtraCalib_p_k,
    unsigned int PinholeExtraCalib_p_k_num_alloc,
    const double* const beta,
    double* out_PinholeExtraCalib_p_kp1,
    unsigned int out_PinholeExtraCalib_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar