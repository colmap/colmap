#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateStep(
    double* ThinPrismFisheyeFocalAndExtra_step_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    const double* const alpha,
    double* out_ThinPrismFisheyeFocalAndExtra_step_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar