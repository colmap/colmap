#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_Mp(double* PinholeFocal_r_k,
                            unsigned int PinholeFocal_r_k_num_alloc,
                            double* PinholeFocal_Mp,
                            unsigned int PinholeFocal_Mp_num_alloc,
                            const double* const beta,
                            double* out_PinholeFocal_Mp_kp1,
                            unsigned int out_PinholeFocal_Mp_kp1_num_alloc,
                            double* out_PinholeFocal_w,
                            unsigned int out_PinholeFocal_w_num_alloc,
                            size_t problem_size);

}  // namespace caspar