#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_retract(
    double* PinholeFocalAndExtra,
    unsigned int PinholeFocalAndExtra_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_PinholeFocalAndExtra_retracted,
    unsigned int out_PinholeFocalAndExtra_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar