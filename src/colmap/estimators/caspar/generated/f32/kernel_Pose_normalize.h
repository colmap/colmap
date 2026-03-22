#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_normalize(float* precond_diag,
                    unsigned int precond_diag_num_alloc,
                    float* precond_tril,
                    unsigned int precond_tril_num_alloc,
                    float* njtr,
                    unsigned int njtr_num_alloc,
                    const float* const diag,
                    float* out_normalized,
                    unsigned int out_normalized_num_alloc,
                    size_t problem_size);

}  // namespace caspar