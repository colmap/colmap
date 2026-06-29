#include "kernel_pinhole_split_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointJtjnjtrDirectKernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        double* focal_jac,
        unsigned int focal_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
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

  __shared__ SharedIndex focal_njtr_indices_loc[1024];
  focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_njtr_indices[global_thread_idx]
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
  LoadShared<2, double, double>(focal_njtr,
                                0 * focal_njtr_num_alloc,
                                focal_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r4, r5);
    r3 = r3 * r5;
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r7, r8);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r11, r12);
    r13 = fma(r10, r12, r6 * r8);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r14, r15);
    r13 = fma(r9, r15, r13);
    r16 = r3 + r13;
    r2 = r2 * r4;
    r10 = fma(r10, r11, r6 * r7);
    r10 = fma(r9, r14, r10);
    r9 = r2 + r10;
    r6 = fma(r0, r9, r1 * r16);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r17, r9, r18 * r16);
    WriteSum2<double, double>((double*)inout_shared, r6, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r19, r6);
    r20 = fma(r19, r9, r6 * r16);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r23 = fma(r21, r9, r22 * r16);
    WriteSum2<double, double>((double*)inout_shared, r20, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r23, r20);
    r24 = fma(r23, r9, r20 * r16);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r25, r26);
    r9 = fma(r25, r9, r26 * r16);
    WriteSum2<double, double>((double*)inout_shared, r24, r9);
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
                        r9,
                        r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r9, r23, r24 * r25);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r25,
                        r16);
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
    r23 = fma(r16, r21, r23);
    r23 = fma(r27, r0, r23);
    r23 = fma(r28, r17, r23);
    r10 = r23 + r10;
    r10 = r4 * r10;
    r22 = fma(r16, r22, r24 * r26);
    r22 = fma(r25, r6, r22);
    r22 = fma(r27, r1, r22);
    r22 = fma(r28, r18, r22);
    r22 = fma(r9, r20, r22);
    r13 = r22 + r13;
    r13 = r5 * r13;
    WriteSum2<double, double>((double*)inout_shared, r10, r13);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r2 + r23;
    r22 = r3 + r22;
    r15 = fma(r15, r22, r14 * r23);
    r12 = fma(r12, r22, r11 * r23);
    WriteSum2<double, double>((double*)inout_shared, r15, r12);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r8, r22, r7 * r23);
    WriteSum1<double, double>((double*)inout_shared, r22);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPrincipalPointJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    double* focal_jac,
    unsigned int focal_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      focal_njtr,
      focal_njtr_num_alloc,
      focal_njtr_indices,
      focal_jac,
      focal_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar