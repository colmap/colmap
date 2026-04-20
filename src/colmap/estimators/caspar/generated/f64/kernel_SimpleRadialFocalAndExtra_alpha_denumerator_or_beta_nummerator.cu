#include "kernel_SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator_kernel(
        double* SimpleRadialFocalAndExtra_p_kp1,
        unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        double* SimpleRadialFocalAndExtra_w,
        unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
        double* const SimpleRadialFocalAndExtra_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialFocalAndExtra_out_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_p_kp1,
        0 * SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_w,
        0 * SimpleRadialFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fma(r1, r3, r0 * r2);
  };
  sum_store<double>(SimpleRadialFocalAndExtra_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      SimpleRadialFocalAndExtra_out_local, SimpleRadialFocalAndExtra_out, 1);
}

void SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    double* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    double* const SimpleRadialFocalAndExtra_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndExtra_alpha_denumerator_or_beta_nummerator_kernel<<<
      n_blocks,
      1024>>>(SimpleRadialFocalAndExtra_p_kp1,
              SimpleRadialFocalAndExtra_p_kp1_num_alloc,
              SimpleRadialFocalAndExtra_w,
              SimpleRadialFocalAndExtra_w_num_alloc,
              SimpleRadialFocalAndExtra_out,
              problem_size);
}

}  // namespace caspar