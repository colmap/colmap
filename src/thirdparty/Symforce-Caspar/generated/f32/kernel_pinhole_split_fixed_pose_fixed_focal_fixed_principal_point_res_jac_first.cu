#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
      r46, r47;

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
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r20 = 2.00000000000000000e+00;
    r21 = fmaf(r11, r18, r14 * r15);
    r22 = r13 * r16;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r12, r17, r21);
    r22 = r20 * r21;
    r23 = r19 * r22;
    r24 = r11 * r17;
    r24 = fmaf(r4, r24, r14 * r16);
    r24 = fmaf(r12, r18, r24);
    r24 = fmaf(r13, r15, r24);
    r25 = -2.00000000000000000e+00;
    r26 = fmaf(r12, r16, r11 * r15);
    r26 = fmaf(r13, r17, r26);
    r26 = fmaf(r4, r26, r14 * r18);
    r18 = r25 * r26;
    r27 = fmaf(r24, r18, r23);
    r7 = fmaf(r8, r27, r7);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r28, r29, r30);
    r31 = r11 * r13;
    r31 = r31 * r20;
    r32 = r12 * r14;
    r33 = fmaf(r25, r32, r31);
    r34 = r11 * r11;
    r34 = r25 * r34;
    r35 = 1.00000000000000000e+00;
    r36 = r12 * r12;
    r36 = fmaf(r25, r36, r35);
    r37 = r34 + r36;
    r38 = r12 * r13;
    r38 = r38 * r20;
    r39 = r11 * r14;
    r39 = fmaf(r20, r39, r38);
    r40 = r20 * r19;
    r40 = r40 * r24;
    r41 = fmaf(r26, r22, r40);
    r42 = r24 * r24;
    r42 = r25 * r42;
    r43 = r35 + r42;
    r44 = r21 * r21;
    r44 = r44 * r25;
    r43 = r43 + r44;
    r7 = fmaf(r28, r33, r7);
    r7 = fmaf(r30, r37, r7);
    r7 = fmaf(r29, r39, r7);
    r7 = fmaf(r9, r41, r7);
    r7 = fmaf(r10, r43, r7);
    r39 = copysign(1.0, r7);
    r39 = fmaf(r0, r39, r7);
    r0 = 1.0 / r39;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r7, r37);
    r42 = r35 + r42;
    r33 = r19 * r19;
    r33 = r25 * r33;
    r42 = r42 + r33;
    r5 = fmaf(r8, r42, r5);
    r22 = r24 * r22;
    r45 = fmaf(r19, r18, r22);
    r46 = r20 * r24;
    r46 = fmaf(r26, r46, r23);
    r32 = fmaf(r20, r32, r31);
    r31 = r13 * r14;
    r23 = r11 * r12;
    r23 = r23 * r20;
    r31 = fmaf(r25, r31, r23);
    r47 = r13 * r13;
    r47 = r25 * r47;
    r36 = r47 + r36;
    r5 = fmaf(r9, r45, r5);
    r5 = fmaf(r10, r46, r5);
    r5 = fmaf(r30, r32, r5);
    r5 = fmaf(r29, r31, r5);
    r5 = fmaf(r28, r36, r5);
    r5 = r7 * r5;
    r2 = fmaf(r0, r5, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r26, r1, r22);
    r8 = fmaf(r8, r1, r6);
    r6 = r13 * r14;
    r6 = fmaf(r20, r6, r23);
    r47 = r35 + r47;
    r47 = r47 + r34;
    r34 = r11 * r14;
    r34 = fmaf(r25, r34, r38);
    r18 = fmaf(r21, r18, r40);
    r33 = r35 + r33;
    r33 = r33 + r44;
    r8 = fmaf(r28, r6, r8);
    r8 = fmaf(r29, r47, r8);
    r8 = fmaf(r30, r34, r8);
    r8 = fmaf(r10, r18, r8);
    r8 = fmaf(r9, r33, r8);
    r8 = r37 * r8;
    r3 = fmaf(r0, r8, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r9 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r9);
  if (global_thread_idx < problem_size) {
    r9 = r4 * r3;
    r10 = r37 * r1;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r39 = r4 * r39;
    r34 = r27 * r39;
    r34 = fmaf(r8, r34, r0 * r10);
    r10 = r4 * r2;
    r5 = r39 * r5;
    r30 = r7 * r42;
    r30 = fmaf(r0, r30, r27 * r5);
    r10 = fmaf(r30, r10, r34 * r9);
    r9 = r4 * r3;
    r47 = r41 * r39;
    r29 = r37 * r33;
    r29 = fmaf(r0, r29, r8 * r47);
    r47 = r4 * r2;
    r6 = r7 * r45;
    r6 = fmaf(r0, r6, r41 * r5);
    r47 = fmaf(r6, r47, r29 * r9);
    r9 = r4 * r3;
    r28 = r37 * r18;
    r44 = r43 * r39;
    r44 = fmaf(r8, r44, r0 * r28);
    r28 = r4 * r2;
    r8 = r7 * r46;
    r5 = fmaf(r43, r5, r0 * r8);
    r28 = fmaf(r5, r28, r44 * r9);
    WriteSum3<float, float>((float*)inout_shared, r10, r47, r28);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r30, r30, r34 * r34);
    r47 = fmaf(r29, r29, r6 * r6);
    r10 = fmaf(r44, r44, r5 * r5);
    WriteSum3<float, float>((float*)inout_shared, r28, r47, r10);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r30, r6, r34 * r29);
    r30 = fmaf(r30, r5, r34 * r44);
    r5 = fmaf(r6, r5, r29 * r44);
    WriteSum3<float, float>((float*)inout_shared, r10, r30, r5);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal,
              focal_num_alloc,
              principal_point,
              principal_point_num_alloc,
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