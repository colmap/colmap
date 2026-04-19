#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_update_r_first(
    double* PinholeFocalAndExtra_r_k,
    unsigned int PinholeFocalAndExtra_r_k_num_alloc,
    double* PinholeFocalAndExtra_w,
    unsigned int PinholeFocalAndExtra_w_num_alloc,
    const double* const negalpha,
    double* out_PinholeFocalAndExtra_r_kp1,
    unsigned int out_PinholeFocalAndExtra_r_kp1_num_alloc,
    double* const out_PinholeFocalAndExtra_r_0_norm2_tot,
    double* const out_PinholeFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar