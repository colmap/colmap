#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_alpha_numerator_denominator(
    float* PinholeFocalAndExtra_p_kp1,
    unsigned int PinholeFocalAndExtra_p_kp1_num_alloc,
    float* PinholeFocalAndExtra_r_k,
    unsigned int PinholeFocalAndExtra_r_k_num_alloc,
    float* PinholeFocalAndExtra_w,
    unsigned int PinholeFocalAndExtra_w_num_alloc,
    float* const PinholeFocalAndExtra_total_ag,
    float* const PinholeFocalAndExtra_total_ac,
    size_t problem_size);

}  // namespace caspar