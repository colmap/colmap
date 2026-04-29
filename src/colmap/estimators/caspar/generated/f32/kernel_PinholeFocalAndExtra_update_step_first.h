#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtraUpdateStepFirst(
    float* PinholeFocalAndExtra_p_kp1,
    unsigned int PinholeFocalAndExtra_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholeFocalAndExtra_step_kp1,
    unsigned int out_PinholeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar