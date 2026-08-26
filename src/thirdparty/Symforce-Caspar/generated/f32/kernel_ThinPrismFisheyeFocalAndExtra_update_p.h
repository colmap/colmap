#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateP(
    float* ThinPrismFisheyeFocalAndExtra_z,
    unsigned int ThinPrismFisheyeFocalAndExtra_z_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_p_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_k_num_alloc,
    const float* const beta,
    float* out_ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar