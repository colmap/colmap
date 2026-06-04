#include "kernel_pinhole_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;
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
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11, r12);
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
    r29 = fmaf(r10, r29, r9);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r30, r31);
    r32 = r13 * r15;
    r32 = r32 * r22;
    r33 = r14 * r16;
    r34 = fmaf(r27, r33, r32);
    r35 = 1.00000000000000000e+00;
    r36 = r14 * r14;
    r36 = r36 * r27;
    r37 = r35 + r36;
    r38 = r13 * r13;
    r38 = r27 * r38;
    r37 = r37 + r38;
    r39 = r14 * r15;
    r39 = r39 * r22;
    r40 = r13 * r16;
    r40 = fmaf(r22, r40, r39);
    r41 = r22 * r21;
    r41 = r41 * r26;
    r42 = fmaf(r28, r24, r41);
    r43 = r23 * r23;
    r43 = r43 * r27;
    r44 = r26 * r26;
    r44 = fmaf(r27, r44, r35);
    r45 = r43 + r44;
    r29 = fmaf(r9, r34, r29);
    r29 = fmaf(r31, r37, r29);
    r29 = fmaf(r30, r40, r29);
    r29 = fmaf(r11, r42, r29);
    r29 = fmaf(r12, r45, r29);
    r45 = copysign(1.0, r29);
    r45 = fmaf(r2, r45, r29);
    r2 = 1.0 / r45;
    r29 = r21 * r21;
    r29 = r27 * r29;
    r44 = r29 + r44;
    r44 = fmaf(r10, r44, r7);
    r24 = r26 * r24;
    r7 = fmaf(r21, r20, r24);
    r42 = r22 * r26;
    r42 = fmaf(r28, r42, r25);
    r33 = fmaf(r22, r33, r32);
    r32 = r15 * r16;
    r25 = r13 * r14;
    r25 = r25 * r22;
    r32 = fmaf(r27, r32, r25);
    r36 = r35 + r36;
    r40 = r15 * r15;
    r40 = r27 * r40;
    r36 = r36 + r40;
    r44 = fmaf(r11, r7, r44);
    r44 = fmaf(r12, r42, r44);
    r44 = fmaf(r31, r33, r44);
    r44 = fmaf(r30, r32, r44);
    r44 = fmaf(r9, r36, r44);
    r36 = r2 * r44;
    r4 = fmaf(r0, r36, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r22 * r21;
    r3 = fmaf(r28, r3, r24);
    r3 = fmaf(r10, r3, r8);
    r10 = r15 * r16;
    r10 = fmaf(r22, r10, r25);
    r40 = r35 + r40;
    r40 = r40 + r38;
    r38 = r13 * r16;
    r38 = fmaf(r27, r38, r39);
    r20 = fmaf(r23, r20, r41);
    r29 = r35 + r29;
    r29 = r29 + r43;
    r3 = fmaf(r9, r10, r3);
    r3 = fmaf(r30, r40, r3);
    r3 = fmaf(r31, r38, r3);
    r3 = fmaf(r12, r20, r3);
    r3 = fmaf(r11, r29, r3);
    r29 = r1 * r3;
    r5 = fmaf(r2, r29, r5);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r29 = fmaf(r4, r4, r5 * r5);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r29);
  if (global_thread_idx < problem_size) {
    r4 = r6 * r4;
    r29 = r6 * r5;
    r11 = r36 * r4;
    r6 = r6 * r3;
    r6 = r6 * r5;
    r6 = r6 * r2;
    WriteSum4<float, float>((float*)inout_shared, r11, r6, r4, r29);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r44 * r44;
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r44 = r44 * r45;
    r29 = r3 * r3;
    r29 = r45 * r29;
    WriteSum4<float, float>((float*)inout_shared, r44, r29, r35, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = 0.00000000000000000e+00;
    WriteSum4<float, float>((float*)inout_shared, r35, r36, r35, r35);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r3 * r2;
    WriteSum2<float, float>((float*)inout_shared, r2, r35);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPoseFixedPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar