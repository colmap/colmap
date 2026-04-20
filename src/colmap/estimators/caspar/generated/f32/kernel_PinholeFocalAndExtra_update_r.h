#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_r(
    float* PinholeFocalAndExtra_r_k,
    unsigned int PinholeFocalAndExtra_r_k_num_alloc,
    float* PinholeFocalAndExtra_w,
    unsigned int PinholeFocalAndExtra_w_num_alloc,
    const float* const negalpha,
    float* out_PinholeFocalAndExtra_r_kp1,
    unsigned int out_PinholeFocalAndExtra_r_kp1_num_alloc,
    float* const out_PinholeFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar