#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateP(
    double* ThinPrismFisheyeFocalAndExtra_z,
    unsigned int ThinPrismFisheyeFocalAndExtra_z_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_p_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_k_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar