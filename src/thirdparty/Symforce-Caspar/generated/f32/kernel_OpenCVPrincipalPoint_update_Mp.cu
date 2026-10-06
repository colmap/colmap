#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVPrincipalPoint_update_Mp.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVPrincipalPointUpdateMpKernel(
    float *OpenCVPrincipalPoint_r_k,
    unsigned int OpenCVPrincipalPoint_r_k_num_alloc,
    float *OpenCVPrincipalPoint_Mp,
    unsigned int OpenCVPrincipalPoint_Mp_num_alloc, const float *const beta,
    float *out_OpenCVPrincipalPoint_Mp_kp1,
    unsigned int out_OpenCVPrincipalPoint_Mp_kp1_num_alloc,
    float *out_OpenCVPrincipalPoint_w,
    unsigned int out_OpenCVPrincipalPoint_w_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(OpenCVPrincipalPoint_Mp,
                                         0 * OpenCVPrincipalPoint_Mp_num_alloc,
                                         global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(OpenCVPrincipalPoint_r_k,
                                         0 * OpenCVPrincipalPoint_r_k_num_alloc,
                                         global_thread_idx, r2, r3);
  };
  LoadUnique<1, float, float>(beta, 0, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float *)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r0, r4, r2);
    r4 = fmaf(r1, r4, r3);
    WriteIdx2<1024, float, float, float2>(
        out_OpenCVPrincipalPoint_Mp_kp1,
        0 * out_OpenCVPrincipalPoint_Mp_kp1_num_alloc, global_thread_idx, r0,
        r4);
    WriteIdx2<1024, float, float, float2>(
        out_OpenCVPrincipalPoint_w, 0 * out_OpenCVPrincipalPoint_w_num_alloc,
        global_thread_idx, r0, r4);
  };
}

void OpenCVPrincipalPointUpdateMp(
    float *OpenCVPrincipalPoint_r_k,
    unsigned int OpenCVPrincipalPoint_r_k_num_alloc,
    float *OpenCVPrincipalPoint_Mp,
    unsigned int OpenCVPrincipalPoint_Mp_num_alloc, const float *const beta,
    float *out_OpenCVPrincipalPoint_Mp_kp1,
    unsigned int out_OpenCVPrincipalPoint_Mp_kp1_num_alloc,
    float *out_OpenCVPrincipalPoint_w,
    unsigned int out_OpenCVPrincipalPoint_w_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVPrincipalPointUpdateMpKernel<<<n_blocks, 1024>>>(
      OpenCVPrincipalPoint_r_k, OpenCVPrincipalPoint_r_k_num_alloc,
      OpenCVPrincipalPoint_Mp, OpenCVPrincipalPoint_Mp_num_alloc, beta,
      out_OpenCVPrincipalPoint_Mp_kp1,
      out_OpenCVPrincipalPoint_Mp_kp1_num_alloc, out_OpenCVPrincipalPoint_w,
      out_OpenCVPrincipalPoint_w_num_alloc, problem_size);
}

} // namespace caspar