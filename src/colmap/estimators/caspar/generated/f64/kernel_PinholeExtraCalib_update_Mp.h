#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_Mp(
    double* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    double* PinholeExtraCalib_Mp,
    unsigned int PinholeExtraCalib_Mp_num_alloc,
    const double* const beta,
    double* out_PinholeExtraCalib_Mp_kp1,
    unsigned int out_PinholeExtraCalib_Mp_kp1_num_alloc,
    double* out_PinholeExtraCalib_w,
    unsigned int out_PinholeExtraCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar