#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;

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
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r0, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r6,
                                         r7,
                                         r8);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
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
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r15, r20, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r13 * r19;
    r24 = fmaf(r4, r24, r16 * r18);
    r24 = fmaf(r14, r20, r24);
    r24 = fmaf(r15, r17, r24);
    r25 = r24 * r24;
    r25 = fmaf(r12, r25, r23);
    r26 = r22 + r25;
    r26 = fmaf(r9, r26, r6);
    r6 = 2.00000000000000000e+00;
    r27 = fmaf(r13, r20, r16 * r17);
    r28 = r15 * r18;
    r27 = fmaf(r4, r28, r27);
    r27 = fmaf(r14, r19, r27);
    r28 = r6 * r27;
    r29 = r24 * r28;
    r30 = fmaf(r14, r18, r13 * r17);
    r30 = fmaf(r15, r19, r30);
    r30 = fmaf(r4, r30, r16 * r20);
    r20 = r12 * r30;
    r31 = fmaf(r21, r20, r29);
    r32 = r6 * r24;
    r33 = r21 * r28;
    r32 = fmaf(r30, r32, r33);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r34, r35, r36);
    r37 = r13 * r15;
    r37 = r37 * r6;
    r38 = r14 * r16;
    r39 = fmaf(r6, r38, r37);
    r40 = r15 * r16;
    r41 = r13 * r14;
    r41 = r41 * r6;
    r40 = fmaf(r12, r40, r41);
    r42 = r14 * r14;
    r42 = r42 * r12;
    r43 = r23 + r42;
    r44 = r15 * r15;
    r44 = r12 * r44;
    r43 = r43 + r44;
    r26 = fmaf(r10, r31, r26);
    r26 = fmaf(r11, r32, r26);
    r26 = fmaf(r36, r39, r26);
    r26 = fmaf(r35, r40, r26);
    r26 = fmaf(r34, r43, r26);
    r43 = 9.99999999999999955e-07;
    r33 = fmaf(r24, r20, r33);
    r33 = fmaf(r9, r33, r8);
    r38 = fmaf(r12, r38, r37);
    r42 = r23 + r42;
    r37 = r13 * r13;
    r37 = r12 * r37;
    r42 = r42 + r37;
    r8 = r14 * r15;
    r8 = r8 * r6;
    r40 = r13 * r16;
    r40 = fmaf(r6, r40, r8);
    r39 = r6 * r21;
    r39 = r39 * r24;
    r28 = fmaf(r30, r28, r39);
    r32 = r27 * r27;
    r32 = r32 * r12;
    r25 = r32 + r25;
    r33 = fmaf(r34, r38, r33);
    r33 = fmaf(r36, r42, r33);
    r33 = fmaf(r35, r40, r33);
    r33 = fmaf(r10, r28, r33);
    r33 = fmaf(r11, r25, r33);
    r25 = copysign(1.0, r33);
    r25 = fmaf(r43, r25, r33);
    r43 = 1.0 / r25;
    r33 = r26 * r43;
    r2 = fmaf(r0, r33, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r6 * r21;
    r1 = fmaf(r30, r1, r29);
    r1 = fmaf(r9, r1, r7);
    r9 = r15 * r16;
    r9 = fmaf(r6, r9, r41);
    r44 = r23 + r44;
    r44 = r44 + r37;
    r37 = r13 * r16;
    r37 = fmaf(r12, r37, r8);
    r20 = fmaf(r27, r20, r39);
    r22 = r23 + r22;
    r22 = r22 + r32;
    r1 = fmaf(r34, r9, r1);
    r1 = fmaf(r35, r44, r1);
    r1 = fmaf(r36, r37, r1);
    r1 = fmaf(r11, r20, r1);
    r1 = fmaf(r10, r22, r1);
    r22 = r5 * r1;
    r3 = fmaf(r43, r22, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r22 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r22);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    r2 = r2 * r33;
    r4 = r4 * r1;
    r4 = r4 * r3;
    r4 = r4 * r43;
    WriteSum2<float, float>((float*)inout_shared, r2, r4);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r26 * r26;
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r26 = r26 * r25;
    r4 = r1 * r1;
    r4 = r25 * r4;
    WriteSum2<float, float>((float*)inout_shared, r26, r4);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              focal,
              focal_num_alloc,
              focal_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_focal_njtr,
              out_focal_njtr_num_alloc,
              out_focal_precond_diag,
              out_focal_precond_diag_num_alloc,
              out_focal_precond_tril,
              out_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar