#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFixedPoseResJacKernel(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
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

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r7,
                                         r8);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r17, r18, r19, r20);
    r21 = fmaf(r13, r18, r16 * r19);
    r22 = r14 * r17;
    r21 = fmaf(r6, r22, r21);
    r21 = fmaf(r15, r20, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r13 * r19;
    r24 = fmaf(r6, r24, r16 * r18);
    r24 = fmaf(r14, r20, r24);
    r24 = fmaf(r15, r17, r24);
    r25 = r24 * r24;
    r25 = fmaf(r12, r25, r23);
    r26 = r22 + r25;
    r2 = fmaf(r9, r26, r2);
    r27 = r21 * r12;
    r28 = fmaf(r14, r18, r13 * r17);
    r28 = fmaf(r15, r19, r28);
    r28 = fmaf(r6, r28, r16 * r20);
    r29 = 2.00000000000000000e+00;
    r20 = fmaf(r13, r20, r16 * r17);
    r30 = r15 * r18;
    r20 = fmaf(r6, r30, r20);
    r20 = fmaf(r14, r19, r20);
    r30 = r29 * r20;
    r31 = r24 * r30;
    r27 = fmaf(r28, r27, r31);
    r32 = r29 * r24;
    r33 = r21 * r30;
    r32 = fmaf(r28, r32, r33);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r34, r35, r36);
    r37 = r13 * r15;
    r37 = r37 * r29;
    r38 = r14 * r16;
    r39 = fmaf(r29, r38, r37);
    r40 = r15 * r16;
    r41 = r13 * r14;
    r41 = r41 * r29;
    r40 = fmaf(r12, r40, r41);
    r42 = r14 * r14;
    r42 = r42 * r12;
    r43 = r23 + r42;
    r44 = r15 * r15;
    r44 = r12 * r44;
    r43 = r43 + r44;
    r2 = fmaf(r10, r27, r2);
    r2 = fmaf(r11, r32, r2);
    r2 = fmaf(r36, r39, r2);
    r2 = fmaf(r35, r40, r2);
    r2 = fmaf(r34, r43, r2);
    r43 = 9.99999999999999955e-07;
    r40 = r12 * r24;
    r40 = fmaf(r28, r40, r33);
    r8 = fmaf(r9, r40, r8);
    r38 = fmaf(r12, r38, r37);
    r42 = r23 + r42;
    r37 = r13 * r13;
    r37 = r12 * r37;
    r42 = r42 + r37;
    r33 = r14 * r15;
    r33 = r33 * r29;
    r39 = r13 * r16;
    r39 = fmaf(r29, r39, r33);
    r45 = r29 * r21;
    r45 = r45 * r24;
    r30 = fmaf(r28, r30, r45);
    r46 = r20 * r20;
    r46 = r46 * r12;
    r25 = r46 + r25;
    r8 = fmaf(r34, r38, r8);
    r8 = fmaf(r36, r42, r8);
    r8 = fmaf(r35, r39, r8);
    r8 = fmaf(r10, r30, r8);
    r8 = fmaf(r11, r25, r8);
    r39 = copysign(1.0, r8);
    r39 = fmaf(r43, r39, r8);
    r43 = r39 * r39;
    r8 = 1.0 / r43;
    r42 = r2 * r2;
    r42 = r8 * r42;
    r38 = r29 * r21;
    r38 = fmaf(r28, r38, r31);
    r9 = fmaf(r9, r38, r7);
    r7 = r15 * r16;
    r7 = fmaf(r29, r7, r41);
    r44 = r23 + r44;
    r44 = r44 + r37;
    r37 = r13 * r16;
    r37 = fmaf(r12, r37, r33);
    r33 = r20 * r12;
    r33 = fmaf(r28, r33, r45);
    r22 = r23 + r22;
    r22 = r22 + r46;
    r9 = fmaf(r34, r7, r9);
    r9 = fmaf(r35, r44, r9);
    r9 = fmaf(r36, r37, r9);
    r9 = fmaf(r11, r33, r9);
    r9 = fmaf(r10, r22, r9);
    r10 = r9 * r9;
    r11 = r8 * r10;
    r37 = r42 + r11;
    r36 = fmaf(r1, r37, r23);
    r44 = r2 * r36;
    r35 = 1.0 / r39;
    r7 = r0 * r35;
    r4 = fmaf(r7, r44, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r9 * r36;
    r5 = fmaf(r7, r3, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r3 = r2 * r36;
    r3 = r3 * r35;
    r44 = r9 * r36;
    r44 = r44 * r35;
    r34 = r2 * r37;
    r34 = r34 * r7;
    r46 = r9 * r37;
    r46 = r46 * r7;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r44,
                                          r34,
                                          r46);
    r45 = r6 * r4;
    r28 = r6 * r5;
    r41 = r2 * r4;
    r31 = r6 * r36;
    r41 = r41 * r35;
    r47 = r9 * r5;
    r47 = r47 * r35;
    r47 = fmaf(r31, r47, r31 * r41);
    r41 = r6 * r2;
    r41 = r41 * r37;
    r41 = r41 * r4;
    r35 = r6 * r9;
    r35 = r35 * r37;
    r35 = r35 * r5;
    r35 = fmaf(r7, r35, r7 * r41);
    WriteSum4<float, float>((float*)inout_shared, r47, r35, r45, r28);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r36 * r36;
    r28 = fmaf(r42, r28, r11 * r28);
    r45 = r0 * r37;
    r37 = r0 * r37;
    r45 = r45 * r37;
    r45 = fmaf(r42, r45, r11 * r45);
    WriteSum4<float, float>((float*)inout_shared, r28, r45, r23, r23);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r36 * r11;
    r45 = r36 * r42;
    r45 = fmaf(r37, r45, r37 * r23);
    WriteSum4<float, float>((float*)inout_shared, r45, r3, r44, r34);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = 0.00000000000000000e+00;
    WriteSum2<float, float>((float*)inout_shared, r46, r34);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r2 * r8;
    r46 = r40 * r34;
    r31 = r0 * r31;
    r0 = r29 * r38;
    r0 = r0 * r9;
    r44 = r29 * r26;
    r44 = fmaf(r34, r44, r8 * r0);
    r0 = r2 * r2;
    r43 = r39 * r43;
    r43 = 1.0 / r43;
    r43 = r12 * r43;
    r39 = r40 * r43;
    r44 = fmaf(r39, r0, r44);
    r44 = fmaf(r10, r39, r44);
    r1 = r1 * r7;
    r44 = r44 * r1;
    r46 = fmaf(r2, r44, r31 * r46);
    r39 = r26 * r36;
    r46 = fmaf(r7, r39, r46);
    r39 = r38 * r36;
    r0 = r9 * r8;
    r0 = r0 * r31;
    r39 = fmaf(r40, r0, r7 * r39);
    r39 = fmaf(r9, r44, r39);
    r44 = r27 * r36;
    r3 = r30 * r34;
    r3 = fmaf(r31, r3, r7 * r44);
    r44 = r30 * r10;
    r45 = r29 * r27;
    r45 = fmaf(r34, r45, r43 * r44);
    r44 = r29 * r22;
    r44 = r44 * r9;
    r45 = fmaf(r8, r44, r45);
    r23 = r30 * r2;
    r23 = r23 * r2;
    r45 = fmaf(r43, r23, r45);
    r23 = r2 * r45;
    r3 = fmaf(r1, r23, r3);
    r23 = r9 * r45;
    r44 = r22 * r36;
    r44 = fmaf(r7, r44, r1 * r23);
    r44 = fmaf(r30, r0, r44);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r46,
                                          r39,
                                          r3,
                                          r44);
    r23 = r25 * r34;
    r37 = r32 * r36;
    r37 = fmaf(r7, r37, r31 * r23);
    r23 = r29 * r33;
    r23 = r23 * r9;
    r31 = r25 * r10;
    r31 = fmaf(r43, r31, r8 * r23);
    r23 = r25 * r2;
    r23 = r23 * r2;
    r31 = fmaf(r43, r23, r31);
    r43 = r29 * r32;
    r31 = fmaf(r34, r43, r31);
    r43 = r2 * r31;
    r37 = fmaf(r1, r43, r37);
    r43 = r33 * r36;
    r0 = fmaf(r25, r0, r7 * r43);
    r43 = r9 * r31;
    r0 = fmaf(r1, r43, r0);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r37, r0);
    r43 = r6 * r4;
    r1 = r6 * r5;
    r1 = fmaf(r39, r1, r46 * r43);
    r43 = r6 * r5;
    r7 = r6 * r4;
    r7 = fmaf(r3, r7, r44 * r43);
    r43 = r6 * r5;
    r23 = r6 * r4;
    r23 = fmaf(r37, r23, r0 * r43);
    WriteSum3<float, float>((float*)inout_shared, r1, r7, r23);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r46, r46, r39 * r39);
    r7 = fmaf(r44, r44, r3 * r3);
    r1 = fmaf(r37, r37, r0 * r0);
    WriteSum3<float, float>((float*)inout_shared, r23, r7, r1);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r46, r3, r39 * r44);
    r39 = fmaf(r39, r0, r46 * r37);
    r0 = fmaf(r44, r0, r3 * r37);
    WriteSum3<float, float>((float*)inout_shared, r1, r39, r0);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialFixedPoseResJac(float* sensor_from_rig,
                                 unsigned int sensor_from_rig_num_alloc,
                                 float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 float* pixel,
                                 unsigned int pixel_num_alloc,
                                 float* pose,
                                 unsigned int pose_num_alloc,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
                                 float* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 float* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 float* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 float* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
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
  SimpleRadialFixedPoseResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
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