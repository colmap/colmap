#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_step(double* PinholeFocal_step_k,
                              unsigned int PinholeFocal_step_k_num_alloc,
                              double* PinholeFocal_p_kp1,
                              unsigned int PinholeFocal_p_kp1_num_alloc,
                              const double* const alpha,
                              double* out_PinholeFocal_step_kp1,
                              unsigned int out_PinholeFocal_step_kp1_num_alloc,
                              size_t problem_size);

}  // namespace caspar