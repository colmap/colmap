#include "kernel_pinhole_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeJtjnjtrDirectKernel(double* pose_njtr,
                               unsigned int pose_njtr_num_alloc,
                               SharedIndex* pose_njtr_indices,
                               double* pose_jac,
                               unsigned int pose_jac_num_alloc,
                               double* calib_njtr,
                               unsigned int calib_njtr_num_alloc,
                               SharedIndex* calib_njtr_indices,
                               double* calib_jac,
                               unsigned int calib_jac_num_alloc,
                               double* point_njtr,
                               unsigned int point_njtr_num_alloc,
                               SharedIndex* point_njtr_indices,
                               double* point_jac,
                               unsigned int point_jac_num_alloc,
                               double* const out_pose_njtr,
                               unsigned int out_pose_njtr_num_alloc,
                               double* const out_calib_njtr,
                               unsigned int out_calib_njtr_num_alloc,
                               double* const out_point_njtr,
                               unsigned int out_point_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r5,
                        r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r6, r7, r2 * r3);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r10, r11);
    r9 = fma(r5, r10, r9);
  };
  LoadShared<2, double, double>(calib_njtr,
                                2 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r12,
                        r13);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                0 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r14,
                        r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r16, r17);
    r14 = fma(r14, r16, r12);
    r12 = r9 + r14;
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r15 = fma(r15, r17, r13);
    r13 = r6 + r15;
    r5 = fma(r1, r13, r0 * r12);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r2, r18);
    r19 = fma(r18, r13, r2 * r12);
    WriteSum2<double, double>((double*)inout_shared, r5, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r19, r5);
    r20 = fma(r5, r13, r19 * r12);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r23 = fma(r22, r13, r21 * r12);
    WriteSum2<double, double>((double*)inout_shared, r20, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r23, r20);
    r24 = fma(r20, r13, r23 * r12);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r25, r26);
    r12 = fma(r25, r12, r26 * r13);
    WriteSum2<double, double>((double*)inout_shared, r24, r12);
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
                        r12,
                        r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r12, r23, r24 * r25);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r25,
                        r13);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r27,
                        r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r25, r19, r23);
    r23 = fma(r13, r21, r23);
    r23 = fma(r27, r0, r23);
    r23 = fma(r28, r2, r23);
    r9 = r23 + r9;
    r16 = r16 * r9;
    r22 = fma(r13, r22, r24 * r26);
    r22 = fma(r25, r5, r22);
    r22 = fma(r27, r1, r22);
    r22 = fma(r28, r18, r22);
    r22 = fma(r12, r20, r22);
    r6 = r22 + r6;
    r17 = r17 * r6;
    WriteSum2<double, double>((double*)inout_shared, r16, r17);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r9, r6);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r23 + r14;
    r15 = r22 + r15;
    r11 = fma(r11, r15, r10 * r14);
    r8 = fma(r8, r15, r7 * r14);
    WriteSum2<double, double>((double*)inout_shared, r11, r8);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r4, r15, r3 * r14);
    WriteSum1<double, double>((double*)inout_shared, r15);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeJtjnjtrDirect(double* pose_njtr,
                          unsigned int pose_njtr_num_alloc,
                          SharedIndex* pose_njtr_indices,
                          double* pose_jac,
                          unsigned int pose_jac_num_alloc,
                          double* calib_njtr,
                          unsigned int calib_njtr_num_alloc,
                          SharedIndex* calib_njtr_indices,
                          double* calib_jac,
                          unsigned int calib_jac_num_alloc,
                          double* point_njtr,
                          unsigned int point_njtr_num_alloc,
                          SharedIndex* point_njtr_indices,
                          double* point_jac,
                          unsigned int point_jac_num_alloc,
                          double* const out_pose_njtr,
                          unsigned int out_pose_njtr_num_alloc,
                          double* const out_calib_njtr,
                          unsigned int out_calib_njtr_num_alloc,
                          double* const out_point_njtr,
                          unsigned int out_point_njtr_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeJtjnjtrDirectKernel<<<n_blocks, 1024>>>(pose_njtr,
                                                 pose_njtr_num_alloc,
                                                 pose_njtr_indices,
                                                 pose_jac,
                                                 pose_jac_num_alloc,
                                                 calib_njtr,
                                                 calib_njtr_num_alloc,
                                                 calib_njtr_indices,
                                                 calib_jac,
                                                 calib_jac_num_alloc,
                                                 point_njtr,
                                                 point_njtr_num_alloc,
                                                 point_njtr_indices,
                                                 point_jac,
                                                 point_jac_num_alloc,
                                                 out_pose_njtr,
                                                 out_pose_njtr_num_alloc,
                                                 out_calib_njtr,
                                                 out_calib_njtr_num_alloc,
                                                 out_point_njtr,
                                                 out_point_njtr_num_alloc,
                                                 problem_size);
}

}  // namespace caspar