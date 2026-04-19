#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_update_r_first(
    double* SimpleRadialFocalAndExtra_r_k,
    unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
    double* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialFocalAndExtra_r_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_r_kp1_num_alloc,
    double* const out_SimpleRadialFocalAndExtra_r_0_norm2_tot,
    double* const out_SimpleRadialFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar