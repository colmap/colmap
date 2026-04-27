#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_Mp(float* PinholeCalib_r_k,
                            unsigned int PinholeCalib_r_k_num_alloc,
                            float* PinholeCalib_Mp,
                            unsigned int PinholeCalib_Mp_num_alloc,
                            const float* const beta,
                            float* out_PinholeCalib_Mp_kp1,
                            unsigned int out_PinholeCalib_Mp_kp1_num_alloc,
                            float* out_PinholeCalib_w,
                            unsigned int out_PinholeCalib_w_num_alloc,
                            size_t problem_size);

}  // namespace caspar