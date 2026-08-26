#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointJtjnjtrDirectKernel(
        double* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        double* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            0 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
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
    r12 = r0 * r9;
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r5 = r1 * r6;
    WriteSum2<double, double>((double*)inout_shared, r12, r5);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            2 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r5,
                                            r12);
    r2 = fma(r12, r6, r5 * r9);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            4 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r14);
    r15 = fma(r14, r6, r13 * r9);
    WriteSum2<double, double>((double*)inout_shared, r2, r15);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            6 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r2);
    r16 = fma(r2, r6, r15 * r9);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            8 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r17,
                                            r18);
    r19 = fma(r18, r6, r17 * r9);
    WriteSum2<double, double>((double*)inout_shared, r16, r19);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            10 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r19,
                                            r16);
    r20 = fma(r16, r6, r19 * r9);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            12 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r21,
                                            r22);
    r23 = fma(r22, r6, r21 * r9);
    WriteSum2<double, double>((double*)inout_shared, r20, r23);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            14 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r23,
                                            r20);
    r9 = r23 * r9;
    r6 = r20 * r6;
    WriteSum2<double, double>((double*)inout_shared, r9, r6);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                2 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r6,
                        r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                4 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r24,
                        r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fma(r24, r15, r9 * r13);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                0 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r13,
                        r26);
  };
  __syncthreads();
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                8 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r27,
                        r28);
  };
  __syncthreads();
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                6 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r29,
                        r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fma(r13, r0, r15);
    r15 = fma(r6, r5, r15);
    r15 = fma(r27, r23, r15);
    r15 = fma(r25, r17, r15);
    r15 = fma(r30, r21, r15);
    r15 = fma(r29, r19, r15);
    r12 = fma(r6, r12, r9 * r14);
    r12 = fma(r24, r2, r12);
    r12 = fma(r26, r1, r12);
    r12 = fma(r30, r22, r12);
    r12 = fma(r29, r16, r12);
    r12 = fma(r28, r20, r12);
    r12 = fma(r25, r18, r12);
    r11 = fma(r11, r12, r10 * r15);
    r8 = fma(r8, r12, r7 * r15);
    WriteSum2<double, double>((double*)inout_shared, r11, r8);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r4, r12, r3 * r15);
    WriteSum1<double, double>((double*)inout_shared, r12);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointJtjnjtrDirect(
    double* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    double* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(focal_and_extra_njtr,
              focal_and_extra_njtr_num_alloc,
              focal_and_extra_njtr_indices,
              focal_and_extra_jac,
              focal_and_extra_jac_num_alloc,
              point_njtr,
              point_njtr_num_alloc,
              point_njtr_indices,
              point_jac,
              point_jac_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              out_point_njtr,
              out_point_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar