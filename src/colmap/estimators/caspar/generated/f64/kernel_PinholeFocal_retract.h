#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_retract(double* PinholeFocal,
                          unsigned int PinholeFocal_num_alloc,
                          double* delta,
                          unsigned int delta_num_alloc,
                          double* out_PinholeFocal_retracted,
                          unsigned int out_PinholeFocal_retracted_num_alloc,
                          size_t problem_size);

}  // namespace caspar