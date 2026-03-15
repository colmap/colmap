#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_p(double* PinholeCalib_z,
                           unsigned int PinholeCalib_z_num_alloc,
                           double* PinholeCalib_p_k,
                           unsigned int PinholeCalib_p_k_num_alloc,
                           const double* const beta,
                           double* out_PinholeCalib_p_kp1,
                           unsigned int out_PinholeCalib_p_kp1_num_alloc,
                           size_t problem_size);

}  // namespace caspar