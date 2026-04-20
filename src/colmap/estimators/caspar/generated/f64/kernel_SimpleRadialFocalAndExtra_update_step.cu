#include "kernel_SimpleRadialFocalAndExtra_update_step.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndExtra_update_step_kernel(
        double* SimpleRadialFocalAndExtra_step_k,
        unsigned int SimpleRadialFocalAndExtra_step_k_num_alloc,
        double* SimpleRadialFocalAndExtra_p_kp1,
        unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        const double* const alpha,
        double* out_SimpleRadialFocalAndExtra_step_kp1,
        unsigned int out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_step_k,
        0 * SimpleRadialFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialFocalAndExtra_p_kp1,
        0 * SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  load_unique<1, double, double>(alpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r4 = fma(r3, r4, r1);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialFocalAndExtra_step_kp1,
        0 * out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r2,
        r4);
  };
}

void SimpleRadialFocalAndExtra_update_step(
    double* SimpleRadialFocalAndExtra_step_k,
    unsigned int SimpleRadialFocalAndExtra_step_k_num_alloc,
    double* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialFocalAndExtra_step_kp1,
    unsigned int out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndExtra_update_step_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocalAndExtra_step_k,
      SimpleRadialFocalAndExtra_step_k_num_alloc,
      SimpleRadialFocalAndExtra_p_kp1,
      SimpleRadialFocalAndExtra_p_kp1_num_alloc,
      alpha,
      out_SimpleRadialFocalAndExtra_step_kp1,
      out_SimpleRadialFocalAndExtra_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar