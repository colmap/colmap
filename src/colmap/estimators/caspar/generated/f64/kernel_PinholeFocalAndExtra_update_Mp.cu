#include "kernel_PinholeFocalAndExtra_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFocalAndExtra_update_Mp_kernel(
        double* PinholeFocalAndExtra_r_k,
        unsigned int PinholeFocalAndExtra_r_k_num_alloc,
        double* PinholeFocalAndExtra_Mp,
        unsigned int PinholeFocalAndExtra_Mp_num_alloc,
        const double* const beta,
        double* out_PinholeFocalAndExtra_Mp_kp1,
        unsigned int out_PinholeFocalAndExtra_Mp_kp1_num_alloc,
        double* out_PinholeFocalAndExtra_w,
        unsigned int out_PinholeFocalAndExtra_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeFocalAndExtra_Mp,
        0 * PinholeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        PinholeFocalAndExtra_r_k,
        0 * PinholeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  load_unique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r4 = fma(r1, r4, r3);
    write_idx_2<1024, double, double, double2>(
        out_PinholeFocalAndExtra_Mp_kp1,
        0 * out_PinholeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r4);
    write_idx_2<1024, double, double, double2>(
        out_PinholeFocalAndExtra_w,
        0 * out_PinholeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r0,
        r4);
  };
}

void PinholeFocalAndExtra_update_Mp(
    double* PinholeFocalAndExtra_r_k,
    unsigned int PinholeFocalAndExtra_r_k_num_alloc,
    double* PinholeFocalAndExtra_Mp,
    unsigned int PinholeFocalAndExtra_Mp_num_alloc,
    const double* const beta,
    double* out_PinholeFocalAndExtra_Mp_kp1,
    unsigned int out_PinholeFocalAndExtra_Mp_kp1_num_alloc,
    double* out_PinholeFocalAndExtra_w,
    unsigned int out_PinholeFocalAndExtra_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFocalAndExtra_update_Mp_kernel<<<n_blocks, 1024>>>(
      PinholeFocalAndExtra_r_k,
      PinholeFocalAndExtra_r_k_num_alloc,
      PinholeFocalAndExtra_Mp,
      PinholeFocalAndExtra_Mp_num_alloc,
      beta,
      out_PinholeFocalAndExtra_Mp_kp1,
      out_PinholeFocalAndExtra_Mp_kp1_num_alloc,
      out_PinholeFocalAndExtra_w,
      out_PinholeFocalAndExtra_w_num_alloc,
      problem_size);
}

}  // namespace caspar