#pragma once

#include "cuda_to_hip.h"

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndExtraUpdateStepFirst(
    float *SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    const float *const alpha, float *out_SimpleRadialFocalAndExtra_step_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

} // namespace caspar