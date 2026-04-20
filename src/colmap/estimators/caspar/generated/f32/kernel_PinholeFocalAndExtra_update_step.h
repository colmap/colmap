#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_step(
    float* PinholeFocalAndExtra_step_k,
    unsigned int PinholeFocalAndExtra_step_k_num_alloc,
    float* PinholeFocalAndExtra_p_kp1,
    unsigned int PinholeFocalAndExtra_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholeFocalAndExtra_step_kp1,
    unsigned int out_PinholeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar