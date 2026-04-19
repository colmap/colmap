#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_p(
    float* PinholeFocalAndExtra_z,
    unsigned int PinholeFocalAndExtra_z_num_alloc,
    float* PinholeFocalAndExtra_p_k,
    unsigned int PinholeFocalAndExtra_p_k_num_alloc,
    const float* const beta,
    float* out_PinholeFocalAndExtra_p_kp1,
    unsigned int out_PinholeFocalAndExtra_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar