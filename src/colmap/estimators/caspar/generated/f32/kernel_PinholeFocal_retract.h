#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_retract(float* PinholeFocal,
                          unsigned int PinholeFocal_num_alloc,
                          float* delta,
                          unsigned int delta_num_alloc,
                          float* out_PinholeFocal_retracted,
                          unsigned int out_PinholeFocal_retracted_num_alloc,
                          size_t problem_size);

}  // namespace caspar