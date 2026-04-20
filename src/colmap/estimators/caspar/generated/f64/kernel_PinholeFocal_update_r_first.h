#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_r_first(double* PinholeFocal_r_k,
                                 unsigned int PinholeFocal_r_k_num_alloc,
                                 double* PinholeFocal_w,
                                 unsigned int PinholeFocal_w_num_alloc,
                                 const double* const negalpha,
                                 double* out_PinholeFocal_r_kp1,
                                 unsigned int out_PinholeFocal_r_kp1_num_alloc,
                                 double* const out_PinholeFocal_r_0_norm2_tot,
                                 double* const out_PinholeFocal_r_kp1_norm2_tot,
                                 size_t problem_size);

}  // namespace caspar