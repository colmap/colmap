#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_pred_decrease_times_two(
    double* SimpleRadialFocalAndExtra_step,
    unsigned int SimpleRadialFocalAndExtra_step_num_alloc,
    double* SimpleRadialFocalAndExtra_precond_diag,
    unsigned int SimpleRadialFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocalAndExtra_njtr,
    unsigned int SimpleRadialFocalAndExtra_njtr_num_alloc,
    double* const out_SimpleRadialFocalAndExtra_pred_dec,
    size_t problem_size);

}  // namespace caspar