#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateMp(
    float* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_Mp,
    unsigned int ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
    const float* const beta,
    float* out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
    float* out_ThinPrismFisheyeFocalAndExtra_w,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar