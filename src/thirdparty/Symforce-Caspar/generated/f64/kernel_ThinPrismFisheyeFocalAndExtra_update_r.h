#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateR(
    double* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    const double* const negalpha,
    double* out_ThinPrismFisheyeFocalAndExtra_r_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
    double* const out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar