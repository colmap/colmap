#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_alpha_numerator_denominator(
    double* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    double* SimpleRadialFocalAndExtra_r_k,
    unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
    double* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    double* const SimpleRadialFocalAndExtra_total_ag,
    double* const SimpleRadialFocalAndExtra_total_ac,
    size_t problem_size);

}  // namespace caspar