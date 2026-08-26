#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointScoreKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49;

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
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r0,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra,
                              8 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12,
                                         r13);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r14, r15, r16);
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r21, r22, r23, r24);
    r25 = fmaf(r17, r22, r20 * r23);
    r26 = r18 * r21;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r19, r24, r25);
    r26 = 2.00000000000000000e+00;
    r27 = fmaf(r17, r24, r20 * r21);
    r28 = r19 * r22;
    r27 = fmaf(r4, r28, r27);
    r27 = fmaf(r18, r23, r27);
    r28 = r26 * r27;
    r29 = r25 * r28;
    r30 = r17 * r23;
    r30 = fmaf(r4, r30, r20 * r22);
    r30 = fmaf(r18, r24, r30);
    r30 = fmaf(r19, r21, r30);
    r31 = -2.00000000000000000e+00;
    r32 = fmaf(r18, r22, r17 * r21);
    r32 = fmaf(r19, r23, r32);
    r32 = fmaf(r4, r32, r20 * r24);
    r24 = r31 * r32;
    r33 = fmaf(r30, r24, r29);
    r33 = fmaf(r14, r33, r13);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r13, r34, r35);
    r36 = r17 * r19;
    r36 = r36 * r26;
    r37 = r18 * r20;
    r38 = fmaf(r31, r37, r36);
    r39 = 1.00000000000000000e+00;
    r40 = r18 * r18;
    r40 = r40 * r31;
    r41 = r39 + r40;
    r42 = r17 * r17;
    r42 = r31 * r42;
    r41 = r41 + r42;
    r43 = r18 * r19;
    r43 = r43 * r26;
    r44 = r17 * r20;
    r44 = fmaf(r26, r44, r43);
    r45 = r26 * r25;
    r45 = r45 * r30;
    r46 = fmaf(r32, r28, r45);
    r47 = r30 * r30;
    r47 = r31 * r47;
    r48 = r39 + r47;
    r49 = r27 * r27;
    r49 = r49 * r31;
    r48 = r48 + r49;
    r33 = fmaf(r13, r38, r33);
    r33 = fmaf(r35, r41, r33);
    r33 = fmaf(r34, r44, r33);
    r33 = fmaf(r15, r46, r33);
    r33 = fmaf(r16, r48, r33);
    r48 = copysign(1.0, r33);
    r48 = fmaf(r10, r48, r33);
    r33 = r48 * r48;
    r33 = 1.0 / r33;
    r47 = r39 + r47;
    r46 = r25 * r25;
    r46 = r31 * r46;
    r47 = r47 + r46;
    r47 = fmaf(r14, r47, r11);
    r28 = r30 * r28;
    r11 = fmaf(r25, r24, r28);
    r44 = r26 * r30;
    r44 = fmaf(r32, r44, r29);
    r37 = fmaf(r26, r37, r36);
    r36 = r19 * r20;
    r29 = r17 * r18;
    r29 = r29 * r26;
    r36 = fmaf(r31, r36, r29);
    r41 = r19 * r19;
    r41 = fmaf(r31, r41, r39);
    r40 = r40 + r41;
    r47 = fmaf(r15, r11, r47);
    r47 = fmaf(r16, r44, r47);
    r47 = fmaf(r35, r37, r47);
    r47 = fmaf(r34, r36, r47);
    r47 = fmaf(r13, r40, r47);
    r40 = r47 * r47;
    r36 = r26 * r25;
    r36 = fmaf(r32, r36, r28);
    r36 = fmaf(r14, r36, r12);
    r14 = r19 * r20;
    r14 = fmaf(r26, r14, r29);
    r41 = r42 + r41;
    r42 = r17 * r20;
    r42 = fmaf(r31, r42, r43);
    r24 = fmaf(r27, r24, r45);
    r46 = r39 + r46;
    r46 = r46 + r49;
    r36 = fmaf(r13, r14, r36);
    r36 = fmaf(r34, r41, r36);
    r36 = fmaf(r35, r42, r36);
    r36 = fmaf(r16, r24, r36);
    r36 = fmaf(r15, r46, r36);
    r46 = r36 * r36;
    r15 = fmaf(r33, r46, r33 * r40);
    r15 = sqrtf(r15);
    r24 = copysign(1.0, r15);
    r24 = fmaf(r10, r24, r15);
    r10 = r24 * r24;
    r10 = 1.0 / r10;
    r10 = r33 * r10;
    r15 = atanf(r15);
    r33 = r15 * r15;
    r10 = r10 * r33;
    r33 = r10 * r46;
    r16 = r10 * r40;
    r42 = r33 + r16;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r35,
                       r41,
                       r34,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 3.00000000000000000e+00;
    r13 = r13 * r10;
    r40 = fmaf(r40, r13, r33);
    r40 = fmaf(r41, r40, r8 * r42);
    r8 = r35 * r26;
    r8 = r8 * r47;
    r8 = r8 * r36;
    r40 = fmaf(r10, r8, r40);
    r33 = r42 * r42;
    r49 = r42 * r33;
    r49 = fmaf(r34, r49, r6 * r42);
    r34 = r33 * r33;
    r49 = fmaf(r14, r34, r49);
    r49 = fmaf(r7, r33, r49);
    r48 = 1.0 / r48;
    r48 = r15 * r48;
    r24 = 1.0 / r24;
    r48 = r48 * r24;
    r49 = r49 * r48;
    r40 = fmaf(r47, r49, r40);
    r40 = fmaf(r47, r48, r40);
    r2 = fmaf(r0, r40, r2);
    r13 = fmaf(r46, r13, r16);
    r13 = fmaf(r35, r13, r9 * r42);
    r42 = r41 * r26;
    r42 = r42 * r47;
    r42 = r42 * r36;
    r13 = fmaf(r10, r42, r13);
    r13 = fmaf(r36, r49, r13);
    r13 = fmaf(r36, r48, r13);
    r13 = fmaf(r5, r13, r1);
    r13 = fmaf(r3, r4, r13);
    r13 = fmaf(r13, r13, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r13);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointScore(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointScoreKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              focal_and_extra_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_rTr,
              problem_size);
}

}  // namespace caspar