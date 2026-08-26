#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateR(
    float* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    const float* const negalpha,
    float* out_ThinPrismFisheyeFocalAndExtra_r_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
    float* const out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar