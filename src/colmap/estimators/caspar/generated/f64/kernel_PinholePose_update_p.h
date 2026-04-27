#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_p(double* PinholePose_z,
                          unsigned int PinholePose_z_num_alloc,
                          double* PinholePose_p_k,
                          unsigned int PinholePose_p_k_num_alloc,
                          const double* const beta,
                          double* out_PinholePose_p_kp1,
                          unsigned int out_PinholePose_p_kp1_num_alloc,
                          size_t problem_size);

}  // namespace caspar