#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_SimpleRadialFocalAndDistortion_update_r_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndDistortionUpdateRFirstKernel(
        double *SimpleRadialFocalAndDistortion_r_k,
        unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
        double *SimpleRadialFocalAndDistortion_w,
        unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
        const double *const negalpha,
        double *out_SimpleRadialFocalAndDistortion_r_kp1,
        unsigned int out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
        double *const out_SimpleRadialFocalAndDistortion_r_0_norm2_tot,
        double *const out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialFocalAndDistortion_r_0_norm2_tot_local[1];

  __shared__ double out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        SimpleRadialFocalAndDistortion_r_k,
        0 * SimpleRadialFocalAndDistortion_r_k_num_alloc, global_thread_idx, r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        SimpleRadialFocalAndDistortion_w,
        0 * SimpleRadialFocalAndDistortion_w_num_alloc, global_thread_idx, r2,
        r3);
  };
  LoadUnique<1, double, double>(negalpha, 0, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r4 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_SimpleRadialFocalAndDistortion_r_kp1,
        0 * out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
        global_thread_idx, r2, r4);
    r1 = fma(r1, r1, r0 * r0);
  };
  SumStore<double>(out_SimpleRadialFocalAndDistortion_r_0_norm2_tot_local,
                   (double *)inout_shared, 0, global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r4 = fma(r4, r4, r2 * r2);
  };
  SumStore<double>(out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot_local,
                   (double *)inout_shared, 0, global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_SimpleRadialFocalAndDistortion_r_0_norm2_tot_local,
                        out_SimpleRadialFocalAndDistortion_r_0_norm2_tot, 1);
  SumFlushFinal<double>(
      out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot_local,
      out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot, 1);
}

void SimpleRadialFocalAndDistortionUpdateRFirst(
    double *SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    double *SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    const double *const negalpha,
    double *out_SimpleRadialFocalAndDistortion_r_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
    double *const out_SimpleRadialFocalAndDistortion_r_0_norm2_tot,
    double *const out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndDistortionUpdateRFirstKernel<<<n_blocks, 1024>>>(
      SimpleRadialFocalAndDistortion_r_k,
      SimpleRadialFocalAndDistortion_r_k_num_alloc,
      SimpleRadialFocalAndDistortion_w,
      SimpleRadialFocalAndDistortion_w_num_alloc, negalpha,
      out_SimpleRadialFocalAndDistortion_r_kp1,
      out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
      out_SimpleRadialFocalAndDistortion_r_0_norm2_tot,
      out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot, problem_size);
}

} // namespace caspar