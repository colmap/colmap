#include "kernel_pinhole_split_fixed_focal_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPointScoreKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r0, r5);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r6,
                                         r7,
                                         r8);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
    r12 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r21 = fmaf(r14, r17, r15 * r20);
    r22 = r13 * r18;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r16, r19, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r15 * r17;
    r24 = fmaf(r4, r24, r14 * r20);
    r24 = fmaf(r16, r18, r24);
    r24 = fmaf(r13, r19, r24);
    r25 = r24 * r24;
    r25 = fmaf(r12, r25, r23);
    r26 = r22 + r25;
    r26 = fmaf(r9, r26, r6);
    r6 = 2.00000000000000000e+00;
    r27 = fmaf(r16, r17, r13 * r20);
    r28 = r14 * r19;
    r27 = fmaf(r4, r28, r27);
    r27 = fmaf(r15, r18, r27);
    r28 = r6 * r27;
    r29 = r24 * r28;
    r30 = fmaf(r14, r18, r13 * r17);
    r30 = fmaf(r15, r19, r30);
    r30 = fmaf(r4, r30, r16 * r20);
    r16 = r12 * r30;
    r31 = fmaf(r21, r16, r29);
    r32 = r6 * r24;
    r33 = r21 * r28;
    r32 = fmaf(r30, r32, r33);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r34,
                       r35,
                       r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r17 * r19;
    r37 = r37 * r6;
    r38 = r18 * r20;
    r39 = fmaf(r6, r38, r37);
    r40 = r19 * r20;
    r41 = r17 * r18;
    r41 = r41 * r6;
    r40 = fmaf(r12, r40, r41);
    r42 = r18 * r18;
    r42 = r42 * r12;
    r43 = r23 + r42;
    r44 = r19 * r19;
    r44 = r12 * r44;
    r43 = r43 + r44;
    r26 = fmaf(r10, r31, r26);
    r26 = fmaf(r11, r32, r26);
    r26 = fmaf(r36, r39, r26);
    r26 = fmaf(r35, r40, r26);
    r26 = fmaf(r34, r43, r26);
    r43 = r0 * r26;
    r40 = 9.99999999999999955e-07;
    r33 = fmaf(r24, r16, r33);
    r33 = fmaf(r9, r33, r8);
    r38 = fmaf(r12, r38, r37);
    r42 = r23 + r42;
    r37 = r17 * r17;
    r37 = r12 * r37;
    r42 = r42 + r37;
    r8 = r18 * r19;
    r8 = r8 * r6;
    r39 = r17 * r20;
    r39 = fmaf(r6, r39, r8);
    r32 = r6 * r21;
    r32 = r32 * r24;
    r28 = fmaf(r30, r28, r32);
    r31 = r27 * r27;
    r31 = r31 * r12;
    r25 = r31 + r25;
    r33 = fmaf(r34, r38, r33);
    r33 = fmaf(r36, r42, r33);
    r33 = fmaf(r35, r39, r33);
    r33 = fmaf(r10, r28, r33);
    r33 = fmaf(r11, r25, r33);
    r25 = copysign(1.0, r33);
    r25 = fmaf(r40, r25, r33);
    r25 = 1.0 / r25;
    r2 = fmaf(r25, r43, r2);
    r4 = fmaf(r3, r4, r1);
    r3 = r6 * r21;
    r3 = fmaf(r30, r3, r29);
    r3 = fmaf(r9, r3, r7);
    r9 = r19 * r20;
    r9 = fmaf(r6, r9, r41);
    r44 = r23 + r44;
    r44 = r44 + r37;
    r37 = r17 * r20;
    r37 = fmaf(r12, r37, r8);
    r16 = fmaf(r27, r16, r32);
    r22 = r23 + r22;
    r22 = r22 + r31;
    r3 = fmaf(r34, r9, r3);
    r3 = fmaf(r35, r44, r3);
    r3 = fmaf(r36, r37, r3);
    r3 = fmaf(r11, r16, r3);
    r3 = fmaf(r10, r22, r3);
    r22 = r5 * r3;
    r4 = fmaf(r25, r22, r4);
    r4 = fmaf(r4, r4, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPointScore(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPointScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar