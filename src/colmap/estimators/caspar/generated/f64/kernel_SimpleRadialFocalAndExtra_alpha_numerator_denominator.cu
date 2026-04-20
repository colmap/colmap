#include "kernel_SimpleRadialFocalAndExtra_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndExtra_alpha_numerator_denominator_kernel(
        double* SimpleRadialFocalAndExtra_p_kp1,
        unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        double* SimpleRadialFocalAndExtra_r_k,
        unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
        double* SimpleRadialFocalAndExtra_w,
        unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
        double* const SimpleRadialFocalAndExtra_total_ag,
        double* const SimpleRadialFocalAndExtra_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialFocalAndExtra_total_ag_local[1];

  __shared__ double SimpleRadialFocalAndExtra_total_ac_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_p_kp1,
        0 * SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_r_k,
        0 * SimpleRadialFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fma(r0, r2, r1 * r3);
  };
  sum_store<double>(SimpleRadialFocalAndExtra_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_w,
        0 * SimpleRadialFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fma(r1, r3, r0 * r2);
  };
  sum_store<double>(SimpleRadialFocalAndExtra_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(SimpleRadialFocalAndExtra_total_ag_local,
                          SimpleRadialFocalAndExtra_total_ag,
                          1);
  sum_flush_final<double>(SimpleRadialFocalAndExtra_total_ac_local,
                          SimpleRadialFocalAndExtra_total_ac,
                          1);
}

void SimpleRadialFocalAndExtra_alpha_numerator_denominator(
    double* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    double* SimpleRadialFocalAndExtra_r_k,
    unsigned int SimpleRadialFocalAndExtra_r_k_num_alloc,
    double* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    double* const SimpleRadialFocalAndExtra_total_ag,
    double* const SimpleRadialFocalAndExtra_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndExtra_alpha_numerator_denominator_kernel<<<n_blocks,
                                                                 1024>>>(
      SimpleRadialFocalAndExtra_p_kp1,
      SimpleRadialFocalAndExtra_p_kp1_num_alloc,
      SimpleRadialFocalAndExtra_r_k,
      SimpleRadialFocalAndExtra_r_k_num_alloc,
      SimpleRadialFocalAndExtra_w,
      SimpleRadialFocalAndExtra_w_num_alloc,
      SimpleRadialFocalAndExtra_total_ag,
      SimpleRadialFocalAndExtra_total_ac,
      problem_size);
}

}  // namespace caspar