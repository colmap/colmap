#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraUpdateMp(
    double* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_Mp,
    unsigned int ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyeFocalAndExtra_w,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar