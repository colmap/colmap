#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_Mp(float* PinholeFocal_r_k,
                            unsigned int PinholeFocal_r_k_num_alloc,
                            float* PinholeFocal_Mp,
                            unsigned int PinholeFocal_Mp_num_alloc,
                            const float* const beta,
                            float* out_PinholeFocal_Mp_kp1,
                            unsigned int out_PinholeFocal_Mp_kp1_num_alloc,
                            float* out_PinholeFocal_w,
                            unsigned int out_PinholeFocal_w_num_alloc,
                            size_t problem_size);

}  // namespace caspar