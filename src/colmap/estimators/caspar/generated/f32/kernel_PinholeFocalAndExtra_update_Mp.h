#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_Mp(
    float* PinholeFocalAndExtra_r_k,
    unsigned int PinholeFocalAndExtra_r_k_num_alloc,
    float* PinholeFocalAndExtra_Mp,
    unsigned int PinholeFocalAndExtra_Mp_num_alloc,
    const float* const beta,
    float* out_PinholeFocalAndExtra_Mp_kp1,
    unsigned int out_PinholeFocalAndExtra_Mp_kp1_num_alloc,
    float* out_PinholeFocalAndExtra_w,
    unsigned int out_PinholeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar