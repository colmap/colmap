#include "kernel_ThinPrismFisheyePrincipalPoint_update_p.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePrincipalPointUpdatePKernel(
        double* ThinPrismFisheyePrincipalPoint_z,
        unsigned int ThinPrismFisheyePrincipalPoint_z_num_alloc,
        double* ThinPrismFisheyePrincipalPoint_p_k,
        unsigned int ThinPrismFisheyePrincipalPoint_p_k_num_alloc,
        const double* const beta,
        double* out_ThinPrismFisheyePrincipalPoint_p_kp1,
        unsigned int out_ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePrincipalPoint_p_k,
        0 * ThinPrismFisheyePrincipalPoint_p_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePrincipalPoint_z,
        0 * ThinPrismFisheyePrincipalPoint_z_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  LoadUnique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r4 = fma(r1, r4, r3);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyePrincipalPoint_p_kp1,
        0 * out_ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r4);
  };
}

void ThinPrismFisheyePrincipalPointUpdateP(
    double* ThinPrismFisheyePrincipalPoint_z,
    unsigned int ThinPrismFisheyePrincipalPoint_z_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_p_k,
    unsigned int ThinPrismFisheyePrincipalPoint_p_k_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePrincipalPointUpdatePKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePrincipalPoint_z,
      ThinPrismFisheyePrincipalPoint_z_num_alloc,
      ThinPrismFisheyePrincipalPoint_p_k,
      ThinPrismFisheyePrincipalPoint_p_k_num_alloc,
      beta,
      out_ThinPrismFisheyePrincipalPoint_p_kp1,
      out_ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar