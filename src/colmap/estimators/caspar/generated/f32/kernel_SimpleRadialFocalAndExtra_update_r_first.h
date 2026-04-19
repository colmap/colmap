#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_update_r_first(
    float* SimpleRadialFocalAndExtra_r_k,
    unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
    float* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialFocalAndExtra_r_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_r_kp1_num_alloc,
    float* const out_SimpleRadialFocalAndExtra_r_0_norm2_tot,
    float* const out_SimpleRadialFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar