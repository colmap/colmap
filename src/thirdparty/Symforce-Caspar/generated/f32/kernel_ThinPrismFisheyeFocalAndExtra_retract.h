#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraRetract(
    float* ThinPrismFisheyeFocalAndExtra,
    unsigned int ThinPrismFisheyeFocalAndExtra_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyeFocalAndExtra_retracted,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar