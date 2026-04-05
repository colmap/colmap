#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_Mp(double* PinholeCalib_r_k,
                            unsigned int PinholeCalib_r_k_num_alloc,
                            double* PinholeCalib_Mp,
                            unsigned int PinholeCalib_Mp_num_alloc,
                            const double* const beta,
                            double* out_PinholeCalib_Mp_kp1,
                            unsigned int out_PinholeCalib_Mp_kp1_num_alloc,
                            double* out_PinholeCalib_w,
                            unsigned int out_PinholeCalib_w_num_alloc,
                            size_t problem_size);

}  // namespace caspar