#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator(
    float* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    float* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    float* const SimpleRadialFocalAndExtra_out,
    size_t problem_size);

}  // namespace caspar