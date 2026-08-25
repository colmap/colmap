#include "kernel_SphericalPose_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalPoseUpdateMpKernel(double* SphericalPose_r_k,
                                unsigned int SphericalPose_r_k_num_alloc,
                                double* SphericalPose_Mp,
                                unsigned int SphericalPose_Mp_num_alloc,
                                const double* const beta,
                                double* out_SphericalPose_Mp_kp1,
                                unsigned int out_SphericalPose_Mp_kp1_num_alloc,
                                double* out_SphericalPose_w,
                                unsigned int out_SphericalPose_w_num_alloc,
                                size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(SphericalPose_Mp,
                                            0 * SphericalPose_Mp_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            0 * SphericalPose_r_k_num_alloc,
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
    r1 = fma(r1, r4, r3);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_Mp_kp1,
        0 * out_SphericalPose_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(SphericalPose_Mp,
                                            2 * SphericalPose_Mp_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r2);
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            2 * SphericalPose_r_k_num_alloc,
                                            global_thread_idx,
                                            r5,
                                            r6);
    r3 = fma(r3, r4, r5);
    r2 = fma(r2, r4, r6);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_Mp_kp1,
        2 * out_SphericalPose_Mp_kp1_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(SphericalPose_Mp,
                                            4 * SphericalPose_Mp_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r5);
    ReadIdx2<1024, double, double, double2>(SphericalPose_r_k,
                                            4 * SphericalPose_r_k_num_alloc,
                                            global_thread_idx,
                                            r7,
                                            r8);
    r6 = fma(r6, r4, r7);
    r4 = fma(r5, r4, r8);
    WriteIdx2<1024, double, double, double2>(
        out_SphericalPose_Mp_kp1,
        4 * out_SphericalPose_Mp_kp1_num_alloc,
        global_thread_idx,
        r6,
        r4);
    WriteIdx2<1024, double, double, double2>(out_SphericalPose_w,
                                             0 * out_SphericalPose_w_num_alloc,
                                             global_thread_idx,
                                             r0,
                                             r1);
    WriteIdx2<1024, double, double, double2>(out_SphericalPose_w,
                                             2 * out_SphericalPose_w_num_alloc,
                                             global_thread_idx,
                                             r3,
                                             r2);
    WriteIdx2<1024, double, double, double2>(out_SphericalPose_w,
                                             4 * out_SphericalPose_w_num_alloc,
                                             global_thread_idx,
                                             r6,
                                             r4);
  };
}

void SphericalPoseUpdateMp(double* SphericalPose_r_k,
                           unsigned int SphericalPose_r_k_num_alloc,
                           double* SphericalPose_Mp,
                           unsigned int SphericalPose_Mp_num_alloc,
                           const double* const beta,
                           double* out_SphericalPose_Mp_kp1,
                           unsigned int out_SphericalPose_Mp_kp1_num_alloc,
                           double* out_SphericalPose_w,
                           unsigned int out_SphericalPose_w_num_alloc,
                           size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalPoseUpdateMpKernel<<<n_blocks, 1024>>>(
      SphericalPose_r_k,
      SphericalPose_r_k_num_alloc,
      SphericalPose_Mp,
      SphericalPose_Mp_num_alloc,
      beta,
      out_SphericalPose_Mp_kp1,
      out_SphericalPose_Mp_kp1_num_alloc,
      out_SphericalPose_w,
      out_SphericalPose_w_num_alloc,
      problem_size);
}

}  // namespace caspar