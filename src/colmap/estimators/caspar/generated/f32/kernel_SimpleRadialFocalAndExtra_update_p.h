#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_update_p(
    float* SimpleRadialFocalAndExtra_z,
    unsigned int SimpleRadialFocalAndExtra_z_num_alloc,
    float* SimpleRadialFocalAndExtra_p_k,
    unsigned int SimpleRadialFocalAndExtra_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocalAndExtra_p_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar