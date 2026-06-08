#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPointJtjnjtrDirectKernel(double* pose_njtr,
                                         unsigned int pose_njtr_num_alloc,
                                         SharedIndex* pose_njtr_indices,
                                         double* pose_jac,
                                         unsigned int pose_jac_num_alloc,
                                         double* calib_njtr,
                                         unsigned int calib_njtr_num_alloc,
                                         SharedIndex* calib_njtr_indices,
                                         double* calib_jac,
                                         unsigned int calib_jac_num_alloc,
                                         double* const out_pose_njtr,
                                         unsigned int out_pose_njtr_num_alloc,
                                         double* const out_calib_njtr,
                                         unsigned int out_calib_njtr_num_alloc,
                                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(calib_njtr,
                                2 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                0 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r6, r7);
    r4 = fma(r4, r6, r2);
    r5 = fma(r5, r7, r3);
    r3 = fma(r1, r5, r0 * r4);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r2, r8);
    r9 = fma(r8, r5, r2 * r4);
    WriteSum2<double, double>((double*)inout_shared, r3, r9);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r9, r3);
    r10 = fma(r3, r5, r9 * r4);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r11, r12);
    r13 = fma(r12, r5, r11 * r4);
    WriteSum2<double, double>((double*)inout_shared, r10, r13);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r13, r10);
    r14 = fma(r10, r5, r13 * r4);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r15, r16);
    r4 = fma(r15, r4, r16 * r5);
    WriteSum2<double, double>((double*)inout_shared, r14, r4);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(pose_njtr,
                                4 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r4,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = fma(r4, r13, r14 * r15);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r15,
                        r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r17,
                        r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = fma(r15, r9, r13);
    r13 = fma(r5, r11, r13);
    r13 = fma(r17, r0, r13);
    r13 = fma(r18, r2, r13);
    r6 = r6 * r13;
    r12 = fma(r5, r12, r14 * r16);
    r12 = fma(r15, r3, r12);
    r12 = fma(r17, r1, r12);
    r12 = fma(r18, r8, r12);
    r12 = fma(r4, r10, r12);
    r7 = r7 * r12;
    WriteSum2<double, double>((double*)inout_shared, r6, r7);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r13, r12);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedPointJtjnjtrDirect(double* pose_njtr,
                                    unsigned int pose_njtr_num_alloc,
                                    SharedIndex* pose_njtr_indices,
                                    double* pose_jac,
                                    unsigned int pose_jac_num_alloc,
                                    double* calib_njtr,
                                    unsigned int calib_njtr_num_alloc,
                                    SharedIndex* calib_njtr_indices,
                                    double* calib_jac,
                                    unsigned int calib_jac_num_alloc,
                                    double* const out_pose_njtr,
                                    unsigned int out_pose_njtr_num_alloc,
                                    double* const out_calib_njtr,
                                    unsigned int out_calib_njtr_num_alloc,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar