#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_update_Mp(
    float* SimpleRadialFocalAndExtra_r_k,
    unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
    float* SimpleRadialFocalAndExtra_Mp,
    unsigned int SimpleRadialFocalAndExtra_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocalAndExtra_Mp_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_Mp_kp1_num_alloc,
    float* out_SimpleRadialFocalAndExtra_w,
    unsigned int out_SimpleRadialFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar