#include "kernel_PinholeExtraCalib_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeExtraCalib_update_Mp_kernel(
    float* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    float* PinholeExtraCalib_Mp,
    unsigned int PinholeExtraCalib_Mp_num_alloc,
    const float* const beta,
    float* out_PinholeExtraCalib_Mp_kp1,
    unsigned int out_PinholeExtraCalib_Mp_kp1_num_alloc,
    float* out_PinholeExtraCalib_w,
    unsigned int out_PinholeExtraCalib_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(PinholeExtraCalib_Mp,
                                           0 * PinholeExtraCalib_Mp_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(PinholeExtraCalib_r_k,
                                           0 * PinholeExtraCalib_r_k_num_alloc,
                                           global_thread_idx,
                                           r2,
                                           r3);
  };
  load_unique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r0, r4, r2);
    r4 = fmaf(r1, r4, r3);
    write_idx_2<1024, float, float, float2>(
        out_PinholeExtraCalib_Mp_kp1,
        0 * out_PinholeExtraCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r4);
    write_idx_2<1024, float, float, float2>(
        out_PinholeExtraCalib_w,
        0 * out_PinholeExtraCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r4);
  };
}

void PinholeExtraCalib_update_Mp(
    float* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    float* PinholeExtraCalib_Mp,
    unsigned int PinholeExtraCalib_Mp_num_alloc,
    const float* const beta,
    float* out_PinholeExtraCalib_Mp_kp1,
    unsigned int out_PinholeExtraCalib_Mp_kp1_num_alloc,
    float* out_PinholeExtraCalib_w,
    unsigned int out_PinholeExtraCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeExtraCalib_update_Mp_kernel<<<n_blocks, 1024>>>(
      PinholeExtraCalib_r_k,
      PinholeExtraCalib_r_k_num_alloc,
      PinholeExtraCalib_Mp,
      PinholeExtraCalib_Mp_num_alloc,
      beta,
      out_PinholeExtraCalib_Mp_kp1,
      out_PinholeExtraCalib_Mp_kp1_num_alloc,
      out_PinholeExtraCalib_w,
      out_PinholeExtraCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar