#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvFixedPoseFixedPointResJacFirstKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
        float *pixel, unsigned int pixel_num_alloc, float *pose,
        unsigned int pose_num_alloc, float *point, unsigned int point_num_alloc,
        float *out_res, unsigned int out_res_num_alloc, float *const out_rTr,
        float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
        float *const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float *const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;
  LoadShared<4, float, float>(calib, 4 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r4, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r11, r12, r13);
    r14 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r20, r18 * r21);
    r24 = r16 * r19;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r22, r23);
    r24 = r23 * r23;
    r24 = r14 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = r15 * r21;
    r27 = fmaf(r25, r27, r18 * r20);
    r27 = fmaf(r16, r22, r27);
    r27 = fmaf(r17, r19, r27);
    r28 = r27 * r27;
    r28 = fmaf(r14, r28, r26);
    r29 = r24 + r28;
    r29 = fmaf(r11, r29, r8);
    r8 = 2.00000000000000000e+00;
    r30 = fmaf(r15, r22, r18 * r19);
    r31 = r17 * r20;
    r30 = fmaf(r25, r31, r30);
    r30 = fmaf(r16, r21, r30);
    r31 = r8 * r30;
    r31 = r31 * r27;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r25, r32, r18 * r22);
    r22 = r14 * r32;
    r33 = fmaf(r23, r22, r31);
    r34 = r8 * r23;
    r34 = r34 * r30;
    r35 = r8 * r27;
    r35 = fmaf(r32, r35, r34);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r36, r37, r38);
    r39 = r15 * r17;
    r39 = r39 * r8;
    r40 = r16 * r18;
    r41 = fmaf(r8, r40, r39);
    r42 = r17 * r18;
    r43 = r15 * r16;
    r43 = r43 * r8;
    r42 = fmaf(r14, r42, r43);
    r44 = r16 * r16;
    r44 = r44 * r14;
    r45 = r26 + r44;
    r46 = r17 * r17;
    r46 = r14 * r46;
    r45 = r45 + r46;
    r29 = fmaf(r12, r33, r29);
    r29 = fmaf(r13, r35, r29);
    r29 = fmaf(r38, r41, r29);
    r29 = fmaf(r37, r42, r29);
    r29 = fmaf(r36, r45, r29);
    r45 = r29 * r29;
    r42 = 3.00000000000000000e+00;
    r41 = 9.99999999999999955e-07;
    r34 = fmaf(r27, r22, r34);
    r34 = fmaf(r11, r34, r10);
    r40 = fmaf(r14, r40, r39);
    r44 = r26 + r44;
    r39 = r15 * r15;
    r39 = r14 * r39;
    r44 = r44 + r39;
    r10 = r16 * r17;
    r10 = r10 * r8;
    r35 = r15 * r18;
    r35 = fmaf(r8, r35, r10);
    r33 = r8 * r23;
    r33 = r33 * r27;
    r47 = r8 * r30;
    r47 = fmaf(r32, r47, r33);
    r48 = r30 * r30;
    r48 = r14 * r48;
    r28 = r48 + r28;
    r34 = fmaf(r36, r40, r34);
    r34 = fmaf(r38, r44, r34);
    r34 = fmaf(r37, r35, r34);
    r34 = fmaf(r12, r47, r34);
    r34 = fmaf(r13, r28, r34);
    r28 = copysign(1.0, r34);
    r28 = fmaf(r41, r28, r34);
    r41 = r28 * r28;
    r34 = 1.0 / r41;
    r45 = r45 * r42;
    r47 = r8 * r23;
    r47 = fmaf(r32, r47, r31);
    r47 = fmaf(r11, r47, r9);
    r11 = r17 * r18;
    r11 = fmaf(r8, r11, r43);
    r46 = r26 + r46;
    r46 = r46 + r39;
    r39 = r15 * r18;
    r39 = fmaf(r14, r39, r10);
    r22 = fmaf(r30, r22, r33);
    r24 = r26 + r24;
    r24 = r24 + r48;
    r47 = fmaf(r36, r11, r47);
    r47 = fmaf(r37, r46, r47);
    r47 = fmaf(r38, r39, r47);
    r47 = fmaf(r13, r22, r47);
    r47 = fmaf(r12, r24, r47);
    r24 = r47 * r47;
    r24 = r24 * r34;
    r45 = fmaf(r34, r45, r24);
    r12 = 1.0 / r28;
    r22 = fmaf(r29, r12, r1 * r45);
    r13 = r29 * r29;
    r13 = r13 * r34;
    r24 = r24 + r13;
    r39 = r24 * r24;
    r7 = fmaf(r7, r39, r6 * r24);
    r6 = r29 * r7;
    r22 = fmaf(r12, r6, r22);
    r38 = r8 * r34;
    r46 = r0 * r38;
    r37 = r29 * r47;
    r22 = fmaf(r37, r46, r22);
    r2 = fmaf(r4, r22, r2);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r46, r6);
    r2 = fmaf(r46, r25, r2);
    r46 = r42 * r47;
    r46 = r46 * r47;
    r46 = fmaf(r34, r46, r13);
    r13 = fmaf(r47, r12, r0 * r46);
    r11 = r47 * r7;
    r13 = fmaf(r12, r11, r13);
    r36 = r1 * r38;
    r13 = fmaf(r37, r36, r13);
    r3 = fmaf(r5, r13, r3);
    r3 = fmaf(r6, r25, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r6 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r6);
  if (global_thread_idx < problem_size) {
    r6 = r25 * r13;
    r6 = r6 * r3;
    r36 = r25 * r2;
    r11 = r22 * r36;
    r48 = r25 * r24;
    r33 = r5 * r47;
    r48 = r48 * r3;
    r48 = r48 * r12;
    r10 = r24 * r12;
    r43 = r4 * r29;
    r10 = r10 * r43;
    r10 = fmaf(r36, r10, r33 * r48);
    r48 = r25 * r3;
    r9 = r12 * r39;
    r48 = r48 * r33;
    r31 = r4 * r12;
    r31 = r31 * r29;
    r31 = r31 * r39;
    r48 = fmaf(r36, r31, r9 * r48);
    WriteSum4<float, float>((float *)inout_shared, r11, r6, r10, r48);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r25 * r3;
    r10 = r5 * r25;
    r10 = r10 * r46;
    r6 = r14 * r47;
    r6 = r6 * r2;
    r6 = r6 * r34;
    r6 = fmaf(r43, r6, r3 * r10);
    r10 = r14 * r29;
    r10 = r10 * r3;
    r10 = r10 * r34;
    r2 = r4 * r45;
    r2 = fmaf(r36, r2, r33 * r10);
    WriteSum4<float, float>((float *)inout_shared, r6, r2, r36, r48);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r22 * r22;
    r36 = r13 * r13;
    r2 = r47 * r34;
    r6 = r5 * r33;
    r2 = r2 * r39;
    r10 = r29 * r34;
    r11 = r4 * r43;
    r10 = r10 * r39;
    r10 = fmaf(r11, r10, r6 * r2);
    r2 = r47 * r47;
    r2 = r34 * r2;
    r32 = r5 * r5;
    r35 = r24 * r39;
    r2 = r2 * r32;
    r2 = r2 * r35;
    r35 = r29 * r24;
    r35 = r35 * r24;
    r35 = r35 * r34;
    r35 = r35 * r39;
    r35 = fmaf(r11, r35, r24 * r2);
    WriteSum4<float, float>((float *)inout_shared, r48, r36, r10, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r5 * r5;
    r10 = r46 * r46;
    r36 = r47 * r11;
    r48 = 4.00000000000000000e+00;
    r41 = r28 * r41;
    r28 = r28 * r41;
    r28 = 1.0 / r28;
    r28 = r48 * r28;
    r28 = r28 * r37;
    r36 = fmaf(r28, r36, r10 * r35);
    r35 = r4 * r4;
    r35 = r35 * r45;
    r10 = r29 * r6;
    r10 = fmaf(r28, r10, r45 * r35);
    WriteSum4<float, float>((float *)inout_shared, r36, r10, r26, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = 0.00000000000000000e+00;
    r10 = r24 * r22;
    r10 = r10 * r12;
    r10 = r10 * r43;
    r36 = r22 * r31;
    r35 = r47 * r22;
    r35 = r35 * r43;
    r35 = r35 * r38;
    WriteSum4<float, float>((float *)inout_shared, r26, r10, r36, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r4 * r45;
    r35 = r35 * r22;
    r36 = r24 * r13;
    r36 = r36 * r12;
    r36 = r36 * r33;
    WriteSum4<float, float>((float *)inout_shared, r35, r22, r26, r36);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r5 * r46;
    r36 = r36 * r13;
    r22 = r13 * r33;
    r22 = r22 * r9;
    r35 = r5 * r8;
    r35 = r35 * r29;
    r35 = r35 * r34;
    r35 = r35 * r47;
    r10 = r13 * r35;
    WriteSum4<float, float>((float *)inout_shared, r22, r36, r10, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r29 * r24;
    r10 = r10 * r34;
    r10 = r10 * r39;
    r10 = fmaf(r11, r10, r2);
    r2 = r24 * r11;
    r41 = 1.0 / r41;
    r41 = r8 * r41;
    r41 = r41 * r37;
    r37 = r46 * r12;
    r36 = r24 * r6;
    r37 = fmaf(r36, r37, r41 * r2);
    r2 = r24 * r12;
    r22 = r45 * r11;
    r2 = fmaf(r22, r2, r41 * r36);
    WriteSum4<float, float>((float *)inout_shared, r13, r10, r37, r2);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r24 * r12;
    r2 = r2 * r43;
    r37 = r24 * r12;
    r37 = r37 * r33;
    r10 = r39 * r11;
    r13 = r46 * r6;
    r13 = fmaf(r9, r13, r41 * r10);
    r10 = r39 * r6;
    r10 = fmaf(r9, r22, r41 * r10);
    WriteSum4<float, float>((float *)inout_shared, r2, r37, r13, r10);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r33 * r9;
    r43 = r47 * r43;
    r43 = r43 * r38;
    r33 = r29 * r46;
    r33 = r33 * r6;
    r10 = r47 * r38;
    r10 = fmaf(r22, r10, r38 * r33);
    WriteSum4<float, float>((float *)inout_shared, r31, r9, r10, r43);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r5 * r46;
    r10 = r4 * r45;
    WriteSum4<float, float>((float *)inout_shared, r43, r10, r35, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPoseFixedPointResJacFirst(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *pose,
    unsigned int pose_num_alloc, float *point, unsigned int point_num_alloc,
    float *out_res, unsigned int out_res_num_alloc, float *const out_rTr,
    float *const out_calib_njtr, unsigned int out_calib_njtr_num_alloc,
    float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, calib, calib_num_alloc,
      calib_indices, pixel, pixel_num_alloc, pose, pose_num_alloc, point,
      point_num_alloc, out_res, out_res_num_alloc, out_rTr, out_calib_njtr,
      out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, problem_size);
}

} // namespace caspar