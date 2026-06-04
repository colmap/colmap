#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedPrincipalPointResJacKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

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
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6);
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
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12,
                                         r13,
                                         r14);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16, r17, r18);
    r19 = fmaf(r11, r16, r14 * r17);
    r20 = r12 * r15;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r13, r18, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r11 * r17;
    r22 = fmaf(r4, r22, r14 * r16);
    r22 = fmaf(r12, r18, r22);
    r22 = fmaf(r13, r15, r22);
    r23 = r22 * r22;
    r23 = fmaf(r10, r23, r21);
    r24 = r20 + r23;
    r0 = fmaf(r7, r24, r0);
    r25 = r19 * r10;
    r26 = fmaf(r12, r16, r11 * r15);
    r26 = fmaf(r13, r17, r26);
    r26 = fmaf(r4, r26, r14 * r18);
    r27 = 2.00000000000000000e+00;
    r18 = fmaf(r11, r18, r14 * r15);
    r28 = r13 * r16;
    r18 = fmaf(r4, r28, r18);
    r18 = fmaf(r12, r17, r18);
    r28 = r27 * r18;
    r29 = r22 * r28;
    r25 = fmaf(r26, r25, r29);
    r30 = r27 * r22;
    r31 = r19 * r28;
    r30 = fmaf(r26, r30, r31);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r32, r33, r34);
    r35 = r11 * r13;
    r35 = r35 * r27;
    r36 = r12 * r14;
    r37 = fmaf(r27, r36, r35);
    r38 = r13 * r14;
    r39 = r11 * r12;
    r39 = r39 * r27;
    r38 = fmaf(r10, r38, r39);
    r40 = r12 * r12;
    r40 = r40 * r10;
    r41 = r21 + r40;
    r42 = r13 * r13;
    r42 = r10 * r42;
    r41 = r41 + r42;
    r0 = fmaf(r8, r25, r0);
    r0 = fmaf(r9, r30, r0);
    r0 = fmaf(r34, r37, r0);
    r0 = fmaf(r33, r38, r0);
    r0 = fmaf(r32, r41, r0);
  };
  LoadShared<2, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r41,
                       r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = 9.99999999999999955e-07;
    r43 = r10 * r22;
    r43 = fmaf(r26, r43, r31);
    r6 = fmaf(r7, r43, r6);
    r36 = fmaf(r10, r36, r35);
    r40 = r21 + r40;
    r35 = r11 * r11;
    r35 = r10 * r35;
    r40 = r40 + r35;
    r31 = r12 * r13;
    r31 = r31 * r27;
    r44 = r11 * r14;
    r44 = fmaf(r27, r44, r31);
    r45 = r27 * r19;
    r45 = r45 * r22;
    r28 = fmaf(r26, r28, r45);
    r46 = r18 * r18;
    r46 = r46 * r10;
    r23 = r46 + r23;
    r6 = fmaf(r32, r36, r6);
    r6 = fmaf(r34, r40, r6);
    r6 = fmaf(r33, r44, r6);
    r6 = fmaf(r8, r28, r6);
    r6 = fmaf(r9, r23, r6);
    r44 = copysign(1.0, r6);
    r44 = fmaf(r37, r44, r6);
    r37 = r44 * r44;
    r6 = 1.0 / r37;
    r40 = r0 * r0;
    r40 = r6 * r40;
    r36 = r27 * r19;
    r36 = fmaf(r26, r36, r29);
    r7 = fmaf(r7, r36, r5);
    r5 = r13 * r14;
    r5 = fmaf(r27, r5, r39);
    r42 = r21 + r42;
    r42 = r42 + r35;
    r35 = r11 * r14;
    r35 = fmaf(r10, r35, r31);
    r31 = r18 * r10;
    r31 = fmaf(r26, r31, r45);
    r20 = r21 + r20;
    r20 = r20 + r46;
    r7 = fmaf(r32, r5, r7);
    r7 = fmaf(r33, r42, r7);
    r7 = fmaf(r34, r35, r7);
    r7 = fmaf(r9, r31, r7);
    r7 = fmaf(r8, r20, r7);
    r8 = r7 * r7;
    r9 = r6 * r8;
    r35 = r40 + r9;
    r21 = fmaf(r38, r35, r21);
    r34 = r0 * r21;
    r42 = 1.0 / r44;
    r33 = r41 * r42;
    r2 = fmaf(r33, r34, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r7 * r21;
    r3 = fmaf(r33, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r21;
    r1 = r1 * r42;
    r34 = r7 * r21;
    r34 = r34 * r42;
    r5 = r0 * r35;
    r5 = r5 * r33;
    r32 = r7 * r35;
    r32 = r32 * r33;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r34,
                                          r5,
                                          r32);
    r32 = r7 * r3;
    r5 = r4 * r21;
    r32 = r32 * r42;
    r34 = r0 * r2;
    r34 = r34 * r42;
    r34 = fmaf(r5, r34, r5 * r32);
    r32 = r4 * r7;
    r32 = r32 * r35;
    r32 = r32 * r3;
    r42 = r4 * r0;
    r42 = r42 * r35;
    r42 = r42 * r2;
    r42 = fmaf(r33, r42, r33 * r32);
    WriteSum2<float, float>((float*)inout_shared, r34, r42);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r21 * r21;
    r34 = r21 * r40;
    r42 = fmaf(r21, r34, r9 * r42);
    r32 = r41 * r35;
    r35 = r41 * r35;
    r32 = r32 * r35;
    r32 = fmaf(r9, r32, r40 * r32);
    WriteSum2<float, float>((float*)inout_shared, r42, r32);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r21 * r9;
    r32 = fmaf(r35, r32, r35 * r34);
    WriteSum1<float, float>((float*)inout_shared, r32);
  };
  FlushSumShared<1, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r24 * r21;
    r34 = r0 * r6;
    r35 = r43 * r34;
    r5 = r41 * r5;
    r35 = fmaf(r5, r35, r33 * r32);
    r32 = r27 * r36;
    r32 = r32 * r7;
    r41 = r27 * r24;
    r41 = fmaf(r34, r41, r6 * r32);
    r32 = r0 * r0;
    r37 = r44 * r37;
    r37 = 1.0 / r37;
    r37 = r10 * r37;
    r44 = r43 * r37;
    r41 = fmaf(r44, r32, r41);
    r41 = fmaf(r8, r44, r41);
    r38 = r38 * r33;
    r41 = r41 * r38;
    r35 = fmaf(r0, r41, r35);
    r44 = r7 * r6;
    r44 = r44 * r5;
    r32 = r36 * r21;
    r32 = fmaf(r33, r32, r43 * r44);
    r32 = fmaf(r7, r41, r32);
    r41 = r25 * r21;
    r42 = r28 * r34;
    r42 = fmaf(r5, r42, r33 * r41);
    r41 = r28 * r8;
    r40 = r27 * r25;
    r40 = fmaf(r34, r40, r37 * r41);
    r41 = r27 * r20;
    r41 = r41 * r7;
    r40 = fmaf(r6, r41, r40);
    r1 = r28 * r0;
    r1 = r1 * r0;
    r40 = fmaf(r37, r1, r40);
    r1 = r0 * r40;
    r42 = fmaf(r38, r1, r42);
    r1 = r7 * r40;
    r1 = fmaf(r38, r1, r28 * r44);
    r41 = r20 * r21;
    r1 = fmaf(r33, r41, r1);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r35,
                                          r32,
                                          r42,
                                          r1);
    r41 = r27 * r31;
    r41 = r41 * r7;
    r46 = r23 * r8;
    r46 = fmaf(r37, r46, r6 * r41);
    r41 = r23 * r0;
    r41 = r41 * r0;
    r46 = fmaf(r37, r41, r46);
    r37 = r27 * r30;
    r46 = fmaf(r34, r37, r46);
    r37 = r0 * r46;
    r41 = r30 * r21;
    r41 = fmaf(r33, r41, r38 * r37);
    r37 = r23 * r34;
    r41 = fmaf(r5, r37, r41);
    r37 = r31 * r21;
    r44 = fmaf(r23, r44, r33 * r37);
    r37 = r7 * r46;
    r44 = fmaf(r38, r37, r44);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r41,
                                          r44);
    r37 = r4 * r3;
    r38 = r4 * r2;
    r38 = fmaf(r35, r38, r32 * r37);
    r37 = r4 * r2;
    r33 = r4 * r3;
    r33 = fmaf(r1, r33, r42 * r37);
    r37 = r4 * r3;
    r5 = r4 * r2;
    r5 = fmaf(r41, r5, r44 * r37);
    WriteSum3<float, float>((float*)inout_shared, r38, r33, r5);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fmaf(r35, r35, r32 * r32);
    r33 = fmaf(r42, r42, r1 * r1);
    r38 = fmaf(r44, r44, r41 * r41);
    WriteSum3<float, float>((float*)inout_shared, r5, r33, r38);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r35, r42, r32 * r1);
    r35 = fmaf(r35, r41, r32 * r44);
    r44 = fmaf(r1, r44, r42 * r41);
    WriteSum3<float, float>((float*)inout_shared, r38, r35, r44);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedPoseFixedPrincipalPointResJac(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPoseFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
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