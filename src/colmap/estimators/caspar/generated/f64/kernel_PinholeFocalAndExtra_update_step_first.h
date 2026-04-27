#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_step_first(
    double* PinholeFocalAndExtra_p_kp1,
    unsigned int PinholeFocalAndExtra_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeFocalAndExtra_step_kp1,
    unsigned int out_PinholeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar