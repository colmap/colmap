#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_p(float* PinholeFocal_z,
                           unsigned int PinholeFocal_z_num_alloc,
                           float* PinholeFocal_p_k,
                           unsigned int PinholeFocal_p_k_num_alloc,
                           const float* const beta,
                           float* out_PinholeFocal_p_kp1,
                           unsigned int out_PinholeFocal_p_kp1_num_alloc,
                           size_t problem_size);

}  // namespace caspar