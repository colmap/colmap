#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        float* const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        float* const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        float* const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r5, r6);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r11, r12, r13, r14);
    r15 = r13 * r14;
    r16 = 2.00000000000000000e+00;
    r17 = r11 * r16;
    r18 = r12 * r17;
    r19 = fmaf(r10, r15, r18);
    r0 = fmaf(r8, r19, r0);
    r20 = r12 * r14;
    r21 = r13 * r17;
    r20 = fmaf(r16, r20, r21);
    r22 = r13 * r13;
    r22 = r22 * r10;
    r23 = 1.00000000000000000e+00;
    r24 = r12 * r12;
    r24 = fmaf(r10, r24, r23);
    r25 = r22 + r24;
    r0 = fmaf(r9, r20, r0);
    r0 = fmaf(r7, r25, r0);
  };
  LoadShared<2, float, float>(focal_and_distortion,
                              0 * focal_and_distortion_num_alloc,
                              focal_and_distortion_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_distortion_indices_loc[threadIdx.x].target,
                       r26,
                       r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r16, r15, r18);
    r5 = fmaf(r7, r15, r5);
    r13 = r12 * r13;
    r13 = r13 * r16;
    r18 = r11 * r14;
    r18 = fmaf(r10, r18, r13);
    r22 = r23 + r22;
    r28 = r11 * r11;
    r28 = r28 * r10;
    r22 = r22 + r28;
    r5 = fmaf(r9, r18, r5);
    r5 = fmaf(r8, r22, r5);
    r29 = r5 * r5;
    r30 = 9.99999999999999955e-07;
    r17 = fmaf(r14, r17, r13);
    r8 = fmaf(r8, r17, r6);
    r6 = r12 * r14;
    r6 = fmaf(r10, r6, r21);
    r24 = r28 + r24;
    r8 = fmaf(r7, r6, r8);
    r8 = fmaf(r9, r24, r8);
    r9 = copysign(1.0, r8);
    r9 = fmaf(r30, r9, r8);
    r30 = r9 * r9;
    r8 = 1.0 / r30;
    r29 = r29 * r8;
    r7 = r0 * r0;
    r28 = r8 * r7;
    r21 = r29 + r28;
    r23 = fmaf(r27, r21, r23);
    r13 = r0 * r23;
    r31 = 1.0 / r9;
    r32 = r26 * r31;
    r2 = fmaf(r32, r13, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r5 * r23;
    r3 = fmaf(r32, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r23;
    r1 = r1 * r31;
    r13 = r5 * r23;
    r13 = r13 * r31;
    r33 = r0 * r21;
    r33 = r33 * r32;
    r34 = r5 * r21;
    r34 = r34 * r32;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r1,
        r13,
        r33,
        r34);
    r34 = r5 * r3;
    r33 = r4 * r23;
    r34 = r34 * r31;
    r13 = r0 * r2;
    r13 = r13 * r31;
    r13 = fmaf(r33, r13, r33 * r34);
    r34 = r4 * r5;
    r34 = r34 * r21;
    r34 = r34 * r3;
    r31 = r0 * r4;
    r31 = r31 * r21;
    r31 = r31 * r2;
    r31 = fmaf(r32, r31, r32 * r34);
    WriteSum2<float, float>((float*)inout_shared, r13, r31);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_njtr,
                           0 * out_focal_and_distortion_njtr_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r23 * r29;
    r13 = r23 * r23;
    r13 = fmaf(r28, r13, r23 * r31);
    r34 = r26 * r21;
    r21 = r26 * r21;
    r34 = r34 * r21;
    r34 = fmaf(r28, r34, r29 * r34);
    WriteSum2<float, float>((float*)inout_shared, r13, r34);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_precond_diag,
                           0 * out_focal_and_distortion_precond_diag_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r23 * r28;
    r31 = fmaf(r21, r31, r21 * r34);
    WriteSum1<float, float>((float*)inout_shared, r31);
  };
  FlushSumShared<1, float>(out_focal_and_distortion_precond_tril,
                           0 * out_focal_and_distortion_precond_tril_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r0 * r8;
    r33 = r26 * r33;
    r31 = r31 * r33;
    r26 = r6 * r7;
    r30 = r9 * r30;
    r30 = 1.0 / r30;
    r30 = r10 * r30;
    r10 = r16 * r25;
    r10 = r10 * r0;
    r10 = fmaf(r8, r10, r30 * r26);
    r26 = r6 * r5;
    r26 = r26 * r5;
    r10 = fmaf(r30, r26, r10);
    r9 = r16 * r15;
    r21 = r5 * r8;
    r10 = fmaf(r21, r9, r10);
    r9 = r0 * r10;
    r27 = r27 * r32;
    r9 = fmaf(r27, r9, r6 * r31);
    r26 = r25 * r23;
    r9 = fmaf(r32, r26, r9);
    r26 = r5 * r10;
    r34 = r6 * r21;
    r34 = fmaf(r33, r34, r27 * r26);
    r26 = r15 * r23;
    r34 = fmaf(r32, r26, r34);
    r26 = r19 * r23;
    r26 = fmaf(r32, r26, r17 * r31);
    r13 = r16 * r19;
    r13 = r13 * r0;
    r29 = r17 * r30;
    r13 = fmaf(r7, r29, r8 * r13);
    r1 = r5 * r5;
    r13 = fmaf(r29, r1, r13);
    r29 = r16 * r22;
    r13 = fmaf(r21, r29, r13);
    r13 = r13 * r27;
    r26 = fmaf(r0, r13, r26);
    r29 = r17 * r21;
    r13 = fmaf(r5, r13, r33 * r29);
    r29 = r22 * r23;
    r13 = fmaf(r32, r29, r13);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r34,
                                          r26,
                                          r13);
    r29 = r20 * r23;
    r1 = r16 * r20;
    r1 = r1 * r0;
    r35 = r24 * r7;
    r35 = fmaf(r30, r35, r8 * r1);
    r1 = r16 * r18;
    r35 = fmaf(r21, r1, r35);
    r8 = r24 * r5;
    r8 = r8 * r5;
    r35 = fmaf(r30, r8, r35);
    r8 = r0 * r35;
    r8 = fmaf(r27, r8, r32 * r29);
    r8 = fmaf(r24, r31, r8);
    r31 = r18 * r23;
    r29 = r24 * r21;
    r29 = fmaf(r33, r29, r32 * r31);
    r31 = r5 * r35;
    r29 = fmaf(r27, r31, r29);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r8, r29);
    r31 = r4 * r2;
    r27 = r4 * r3;
    r27 = fmaf(r34, r27, r9 * r31);
    r31 = r4 * r2;
    r33 = r4 * r3;
    r33 = fmaf(r13, r33, r26 * r31);
    r31 = r4 * r3;
    r32 = r4 * r2;
    r32 = fmaf(r8, r32, r29 * r31);
    WriteSum3<float, float>((float*)inout_shared, r27, r33, r32);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r9, r9, r34 * r34);
    r33 = fmaf(r13, r13, r26 * r26);
    r27 = fmaf(r8, r8, r29 * r29);
    WriteSum3<float, float>((float*)inout_shared, r32, r33, r27);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r34, r13, r9 * r26);
    r34 = fmaf(r34, r29, r9 * r8);
    r29 = fmaf(r13, r29, r26 * r8);
    WriteSum3<float, float>((float*)inout_shared, r27, r34, r29);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirst(
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    float* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    float* const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    float* const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar