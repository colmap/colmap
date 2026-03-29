#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_p(float* Point_z,
                    unsigned int Point_z_num_alloc,
                    float* Point_p_k,
                    unsigned int Point_p_k_num_alloc,
                    const float* const beta,
                    float* out_Point_p_kp1,
                    unsigned int out_Point_p_kp1_num_alloc,
                    size_t problem_size);

}  // namespace caspar