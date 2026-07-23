#include "kernel_spherical_fixed_pose_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPoseResJacFirstKernel(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* wh,
    unsigned int wh_num_alloc,
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
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r4,
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
    r19 = r11 * r17;
    r20 = -1.00000000000000000e+00;
    r19 = fmaf(r20, r19, r14 * r16);
    r19 = fmaf(r12, r18, r19);
    r19 = fmaf(r13, r15, r19);
    r21 = r10 * r19;
    r21 = r21 * r19;
    r22 = 1.00000000000000000e+00;
    r23 = fmaf(r11, r16, r14 * r17);
    r24 = r12 * r15;
    r23 = fmaf(r20, r24, r23);
    r23 = fmaf(r13, r18, r23);
    r24 = r23 * r23;
    r24 = fmaf(r10, r24, r22);
    r25 = r21 + r24;
    r4 = fmaf(r7, r25, r4);
    r26 = fmaf(r11, r18, r14 * r15);
    r27 = r13 * r16;
    r26 = fmaf(r20, r27, r26);
    r26 = fmaf(r12, r17, r26);
    r27 = 2.00000000000000000e+00;
    r28 = r27 * r19;
    r29 = r26 * r28;
    r30 = fmaf(r12, r16, r11 * r15);
    r30 = fmaf(r13, r17, r30);
    r30 = fmaf(r20, r30, r14 * r18);
    r18 = r10 * r30;
    r31 = fmaf(r23, r18, r29);
    r32 = r23 * r27;
    r32 = r32 * r26;
    r33 = fmaf(r30, r28, r32);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r34, r35, r36);
    r37 = r11 * r13;
    r37 = r37 * r27;
    r38 = r12 * r14;
    r39 = fmaf(r27, r38, r37);
    r40 = r13 * r14;
    r41 = r11 * r12;
    r41 = r41 * r27;
    r40 = fmaf(r10, r40, r41);
    r42 = r13 * r13;
    r42 = r10 * r42;
    r43 = r22 + r42;
    r44 = r12 * r12;
    r44 = r44 * r10;
    r43 = r43 + r44;
    r4 = fmaf(r8, r31, r4);
    r4 = fmaf(r9, r33, r4);
    r4 = fmaf(r36, r39, r4);
    r4 = fmaf(r35, r40, r4);
    r4 = fmaf(r34, r43, r4);
    r43 = 9.99999999999999955e-07;
    r19 = fmaf(r19, r18, r32);
    r6 = fmaf(r7, r19, r6);
    r38 = fmaf(r10, r38, r37);
    r44 = r22 + r44;
    r37 = r11 * r11;
    r37 = r10 * r37;
    r44 = r44 + r37;
    r32 = r12 * r13;
    r32 = r32 * r27;
    r40 = r11 * r14;
    r40 = fmaf(r27, r40, r32);
    r39 = r27 * r26;
    r28 = r23 * r28;
    r39 = fmaf(r30, r39, r28);
    r21 = r22 + r21;
    r45 = r26 * r26;
    r45 = r10 * r45;
    r21 = r21 + r45;
    r6 = fmaf(r34, r38, r6);
    r6 = fmaf(r36, r44, r6);
    r6 = fmaf(r35, r40, r6);
    r6 = fmaf(r8, r39, r6);
    r6 = fmaf(r9, r21, r6);
    r40 = copysignf(r43, r6);
    r40 = r40 + r6;
    r44 = atan2f(r4, r40);
    r44 = fmaf(r3, r44, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r38);
    r3 = fmaf(r3, r20, r0 * r44);
    r44 = -3.18309886183790691e-01;
    r46 = r23 * r27;
    r46 = fmaf(r30, r46, r29);
    r7 = fmaf(r7, r46, r5);
    r5 = r13 * r14;
    r5 = fmaf(r27, r5, r41);
    r42 = r22 + r42;
    r42 = r42 + r37;
    r37 = r11 * r14;
    r37 = fmaf(r10, r37, r32);
    r18 = fmaf(r26, r18, r28);
    r24 = r45 + r24;
    r7 = fmaf(r34, r5, r7);
    r7 = fmaf(r35, r42, r7);
    r7 = fmaf(r36, r37, r7);
    r7 = fmaf(r9, r18, r7);
    r7 = fmaf(r8, r24, r7);
    r8 = r20 * r7;
    r9 = r4 * r4;
    r37 = r43 + r9;
    r37 = fmaf(r6, r6, r37);
    r36 = sqrtf(r37);
    r43 = copysignf(r43, r36);
    r36 = r43 + r36;
    r8 = atan2f(r8, r36);
    r8 = fmaf(r44, r8, r2);
    r38 = fmaf(r38, r20, r1 * r8);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r38);
    r8 = fmaf(r38, r38, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r8);
  if (global_thread_idx < problem_size) {
    r8 = 3.18309886183790691e-01;
    r8 = r38 * r8;
    r38 = r36 * r36;
    r44 = fmaf(r7, r7, r38);
    r43 = 1.0 / r44;
    r42 = r1 * r38;
    r8 = r8 * r43;
    r8 = r8 * r42;
    r43 = r20 * r46;
    r36 = 1.0 / r36;
    r35 = r27 * r25;
    r5 = r27 * r19;
    r5 = fmaf(r6, r5, r4 * r35);
    r7 = r2 * r7;
    r2 = 1.0 / r38;
    r37 = rsqrtf(r37);
    r7 = r7 * r2;
    r7 = r7 * r37;
    r5 = fmaf(r5, r7, r36 * r43);
    r43 = -1.59154943091895346e-01;
    r43 = r3 * r43;
    r3 = r40 * r40;
    r9 = r9 + r3;
    r37 = 1.0 / r9;
    r2 = r0 * r3;
    r43 = r43 * r37;
    r43 = r43 * r2;
    r37 = r20 * r4;
    r35 = 1.0 / r3;
    r37 = r37 * r35;
    r40 = 1.0 / r40;
    r35 = fmaf(r25, r40, r19 * r37);
    r34 = fmaf(r35, r43, r5 * r8);
    r45 = fmaf(r39, r37, r31 * r40);
    r28 = r20 * r24;
    r32 = r27 * r39;
    r10 = r27 * r31;
    r10 = fmaf(r4, r10, r6 * r32);
    r10 = fmaf(r10, r7, r36 * r28);
    r28 = fmaf(r10, r8, r45 * r43);
    r37 = fmaf(r21, r37, r33 * r40);
    r40 = r20 * r18;
    r32 = r27 * r33;
    r22 = r27 * r21;
    r22 = fmaf(r6, r22, r4 * r32);
    r7 = fmaf(r22, r7, r36 * r40);
    r8 = fmaf(r7, r8, r37 * r43);
    WriteSum3<float, float>((float*)inout_shared, r34, r28, r8);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = 1.01321183642337789e-01;
    r8 = r1 * r8;
    r44 = r44 * r44;
    r44 = 1.0 / r44;
    r8 = r8 * r44;
    r8 = r8 * r42;
    r8 = r8 * r38;
    r38 = r5 * r8;
    r42 = 2.53302959105844473e-02;
    r42 = r0 * r42;
    r9 = r9 * r9;
    r9 = 1.0 / r9;
    r42 = r42 * r9;
    r42 = r42 * r2;
    r42 = r42 * r3;
    r3 = r35 * r42;
    r35 = fmaf(r35, r3, r5 * r38);
    r5 = r45 * r45;
    r2 = r10 * r10;
    r2 = fmaf(r8, r2, r42 * r5);
    r5 = r37 * r37;
    r9 = r7 * r7;
    r9 = fmaf(r8, r9, r42 * r5);
    WriteSum3<float, float>((float*)inout_shared, r35, r2, r9);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r45, r3, r10 * r38);
    r3 = fmaf(r37, r3, r7 * r38);
    r38 = r10 * r7;
    r2 = r45 * r37;
    r2 = fmaf(r42, r2, r8 * r38);
    WriteSum3<float, float>((float*)inout_shared, r9, r3, r2);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SphericalFixedPoseResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* wh,
    unsigned int wh_num_alloc,
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
  SphericalFixedPoseResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      wh,
      wh_num_alloc,
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
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar