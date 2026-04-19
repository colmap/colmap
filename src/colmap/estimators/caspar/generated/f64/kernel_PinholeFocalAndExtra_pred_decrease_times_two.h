#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_pred_decrease_times_two(
    double* PinholeFocalAndExtra_step,
    unsigned int PinholeFocalAndExtra_step_num_alloc,
    double* PinholeFocalAndExtra_precond_diag,
    unsigned int PinholeFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeFocalAndExtra_njtr,
    unsigned int PinholeFocalAndExtra_njtr_num_alloc,
    double* const out_PinholeFocalAndExtra_pred_dec,
    size_t problem_size);

}  // namespace caspar