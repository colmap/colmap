#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraRetract(
    double* ThinPrismFisheyeFocalAndExtra,
    unsigned int ThinPrismFisheyeFocalAndExtra_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_ThinPrismFisheyeFocalAndExtra_retracted,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar