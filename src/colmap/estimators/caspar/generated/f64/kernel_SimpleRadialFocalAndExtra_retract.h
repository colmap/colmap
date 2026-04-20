#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_retract(
    double* SimpleRadialFocalAndExtra,
    unsigned int SimpleRadialFocalAndExtra_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialFocalAndExtra_retracted,
    unsigned int out_SimpleRadialFocalAndExtra_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar