#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_update_step_first(
    double* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialFocalAndExtra_step_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar