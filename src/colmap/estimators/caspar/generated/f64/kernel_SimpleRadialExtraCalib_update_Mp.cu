#include "kernel_SimpleRadialExtraCalib_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_update_Mp_kernel(
        double* SimpleRadialExtraCalib_r_k,
        unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
        double* SimpleRadialExtraCalib_Mp,
        unsigned int SimpleRadialExtraCalib_Mp_num_alloc,
        const double* const beta,
        double* out_SimpleRadialExtraCalib_Mp_kp1,
        unsigned int out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
        double* out_SimpleRadialExtraCalib_w,
        unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_Mp,
        0 * SimpleRadialExtraCalib_Mp_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_r_k,
        0 * SimpleRadialExtraCalib_r_k_num_alloc,
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
    r1 = fma(r1, r4, r3);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialExtraCalib_Mp_kp1,
        0 * out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_Mp,
        2 * SimpleRadialExtraCalib_Mp_num_alloc,
        global_thread_idx,
        r3);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_r_k,
        2 * SimpleRadialExtraCalib_r_k_num_alloc,
        global_thread_idx,
        r2);
    r4 = fma(r3, r4, r2);
    write_idx_1<1024, double, double, double>(
        out_SimpleRadialExtraCalib_Mp_kp1,
        2 * out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r4);
    write_idx_2<1024, double, double, double2>(
        out_SimpleRadialExtraCalib_w,
        0 * out_SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    write_idx_1<1024, double, double, double>(
        out_SimpleRadialExtraCalib_w,
        2 * out_SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r4);
  };
}

void SimpleRadialExtraCalib_update_Mp(
    double* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    double* SimpleRadialExtraCalib_Mp,
    unsigned int SimpleRadialExtraCalib_Mp_num_alloc,
    const double* const beta,
    double* out_SimpleRadialExtraCalib_Mp_kp1,
    unsigned int out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
    double* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_update_Mp_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_r_k,
      SimpleRadialExtraCalib_r_k_num_alloc,
      SimpleRadialExtraCalib_Mp,
      SimpleRadialExtraCalib_Mp_num_alloc,
      beta,
      out_SimpleRadialExtraCalib_Mp_kp1,
      out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
      out_SimpleRadialExtraCalib_w,
      out_SimpleRadialExtraCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar