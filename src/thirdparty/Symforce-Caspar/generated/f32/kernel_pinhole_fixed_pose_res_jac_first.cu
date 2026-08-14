#include "kernel_pinhole_fixed_pose_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeFixedPoseResJacFirstKernel(
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
    float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

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
    r2 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r8,
                                         r9);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r22 = 2.00000000000000000e+00;
    r23 = fmaf(r13, r20, r16 * r17);
    r24 = r15 * r18;
    r23 = fmaf(r6, r24, r23);
    r23 = fmaf(r14, r19, r23);
    r24 = r22 * r23;
    r25 = r21 * r24;
    r26 = r13 * r19;
    r26 = fmaf(r6, r26, r16 * r18);
    r26 = fmaf(r14, r20, r26);
    r26 = fmaf(r15, r17, r26);
    r27 = -2.00000000000000000e+00;
    r28 = fmaf(r14, r18, r13 * r17);
    r28 = fmaf(r15, r19, r28);
    r28 = fmaf(r6, r28, r16 * r20);
    r20 = r27 * r28;
    r29 = fmaf(r26, r20, r25);
    r9 = fmaf(r10, r29, r9);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r30, r31, r32);
    r33 = r13 * r15;
    r33 = r33 * r22;
    r34 = r14 * r16;
    r35 = fmaf(r27, r34, r33);
    r36 = r13 * r13;
    r36 = r27 * r36;
    r37 = 1.00000000000000000e+00;
    r38 = r14 * r14;
    r38 = fmaf(r27, r38, r37);
    r39 = r36 + r38;
    r40 = r14 * r15;
    r40 = r40 * r22;
    r41 = r13 * r16;
    r41 = fmaf(r22, r41, r40);
    r42 = r22 * r21;
    r42 = r42 * r26;
    r43 = fmaf(r28, r24, r42);
    r44 = r26 * r26;
    r44 = r27 * r44;
    r45 = r37 + r44;
    r46 = r23 * r23;
    r46 = r46 * r27;
    r45 = r45 + r46;
    r9 = fmaf(r30, r35, r9);
    r9 = fmaf(r32, r39, r9);
    r9 = fmaf(r31, r41, r9);
    r9 = fmaf(r11, r43, r9);
    r9 = fmaf(r12, r45, r9);
    r41 = copysign(1.0, r9);
    r41 = fmaf(r2, r41, r9);
    r2 = 1.0 / r41;
    r44 = r37 + r44;
    r9 = r21 * r21;
    r9 = r27 * r9;
    r44 = r44 + r9;
    r7 = fmaf(r10, r44, r7);
    r24 = r26 * r24;
    r39 = fmaf(r21, r20, r24);
    r35 = r22 * r26;
    r35 = fmaf(r28, r35, r25);
    r34 = fmaf(r22, r34, r33);
    r33 = r15 * r16;
    r25 = r13 * r14;
    r25 = r25 * r22;
    r33 = fmaf(r27, r33, r25);
    r47 = r15 * r15;
    r47 = r27 * r47;
    r38 = r47 + r38;
    r7 = fmaf(r11, r39, r7);
    r7 = fmaf(r12, r35, r7);
    r7 = fmaf(r32, r34, r7);
    r7 = fmaf(r31, r33, r7);
    r7 = fmaf(r30, r38, r7);
    r38 = r0 * r7;
    r4 = fmaf(r2, r38, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r22 * r21;
    r3 = fmaf(r28, r3, r24);
    r10 = fmaf(r10, r3, r8);
    r8 = r15 * r16;
    r8 = fmaf(r22, r8, r25);
    r47 = r37 + r47;
    r47 = r47 + r36;
    r36 = r13 * r16;
    r36 = fmaf(r27, r36, r40);
    r20 = fmaf(r23, r20, r42);
    r9 = r37 + r9;
    r9 = r9 + r46;
    r10 = fmaf(r30, r8, r10);
    r10 = fmaf(r31, r47, r10);
    r10 = fmaf(r32, r36, r10);
    r10 = fmaf(r12, r20, r10);
    r10 = fmaf(r11, r9, r10);
    r11 = r1 * r10;
    r5 = fmaf(r2, r11, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r12 = fmaf(r4, r4, r5 * r5);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r12);
  if (global_thread_idx < problem_size) {
    r12 = r7 * r2;
    r36 = r10 * r2;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r12,
                                          r36);
    r32 = r6 * r4;
    r47 = r6 * r5;
    r31 = r6 * r7;
    r31 = r31 * r4;
    r31 = r31 * r2;
    r8 = r6 * r10;
    r8 = r8 * r5;
    r8 = r8 * r2;
    WriteSum4<float, float>((float*)inout_shared, r31, r8, r32, r47);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r7 * r7;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r7 = r7 * r41;
    r10 = r10 * r10;
    r10 = r10 * r41;
    WriteSum4<float, float>((float*)inout_shared, r7, r10, r37, r37);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = 0.00000000000000000e+00;
    WriteSum4<float, float>((float*)inout_shared, r37, r12, r37, r37);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r36, r37);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r6 * r41;
    r38 = r41 * r38;
    r37 = r0 * r44;
    r37 = fmaf(r2, r37, r29 * r38);
    r36 = r29 * r41;
    r12 = r1 * r3;
    r12 = fmaf(r2, r12, r11 * r36);
    r36 = r0 * r39;
    r36 = fmaf(r2, r36, r43 * r38);
    r10 = r1 * r9;
    r7 = r43 * r41;
    r7 = fmaf(r11, r7, r2 * r10);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r37,
                                          r12,
                                          r36,
                                          r7);
    r10 = r0 * r35;
    r38 = fmaf(r45, r38, r2 * r10);
    r10 = r1 * r20;
    r47 = r45 * r41;
    r47 = fmaf(r11, r47, r2 * r10);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r38,
                                          r47);
    r10 = r6 * r4;
    r11 = r6 * r5;
    r11 = fmaf(r12, r11, r37 * r10);
    r10 = r6 * r5;
    r2 = r6 * r4;
    r2 = fmaf(r36, r2, r7 * r10);
    r10 = r6 * r4;
    r32 = r6 * r5;
    r32 = fmaf(r47, r32, r38 * r10);
    WriteSum3<float, float>((float*)inout_shared, r11, r2, r32);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r37, r37, r12 * r12);
    r2 = fmaf(r7, r7, r36 * r36);
    r11 = fmaf(r38, r38, r47 * r47);
    WriteSum3<float, float>((float*)inout_shared, r32, r2, r11);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r37, r36, r12 * r7);
    r12 = fmaf(r12, r47, r37 * r38);
    r47 = fmaf(r7, r47, r36 * r38);
    WriteSum3<float, float>((float*)inout_shared, r11, r12, r47);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPoseResJacFirst(float* sensor_from_rig,
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
                                 float* const out_rTr,
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
  PinholeFixedPoseResJacFirstKernel<<<n_blocks, 1024>>>(
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
      out_rTr,
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