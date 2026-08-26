#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateStepFirst(
    float* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    const float* const alpha,
    float* out_ThinPrismFisheyeFocalAndExtra_step_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar