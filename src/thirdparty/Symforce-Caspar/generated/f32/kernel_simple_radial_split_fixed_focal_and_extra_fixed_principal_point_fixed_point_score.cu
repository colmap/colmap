#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointScoreKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
    r3 = fmaf(r3, r4, r1);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r1,
                                         r5,
                                         r6);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = r12 * r14;
    r18 = fmaf(r4, r18, r11 * r17);
    r18 = fmaf(r13, r15, r18);
    r18 = fmaf(r10, r16, r18);
    r19 = 2.00000000000000000e+00;
    r20 = fmaf(r13, r14, r10 * r17);
    r21 = r11 * r16;
    r20 = fmaf(r4, r21, r20);
    r20 = fmaf(r12, r15, r20);
    r21 = r19 * r20;
    r22 = r18 * r21;
    r23 = fmaf(r11, r14, r12 * r17);
    r24 = r10 * r15;
    r23 = fmaf(r4, r24, r23);
    r23 = fmaf(r13, r16, r23);
    r24 = fmaf(r11, r15, r10 * r14);
    r24 = fmaf(r12, r16, r24);
    r24 = fmaf(r4, r24, r13 * r17);
    r13 = r23 * r24;
    r25 = fmaf(r19, r13, r22);
    r25 = fmaf(r7, r25, r5);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r5,
                       r26,
                       r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r14 * r15;
    r28 = r28 * r19;
    r29 = r16 * r17;
    r29 = fmaf(r19, r29, r28);
    r30 = -2.00000000000000000e+00;
    r31 = r16 * r16;
    r31 = r30 * r31;
    r32 = 1.00000000000000000e+00;
    r33 = r14 * r14;
    r33 = fmaf(r30, r33, r32);
    r34 = r31 + r33;
    r35 = r15 * r16;
    r35 = r35 * r19;
    r36 = r17 * r30;
    r37 = fmaf(r14, r36, r35);
    r38 = r19 * r23;
    r38 = r38 * r18;
    r39 = r20 * r30;
    r39 = fmaf(r24, r39, r38);
    r40 = r20 * r20;
    r40 = r40 * r30;
    r41 = r32 + r40;
    r42 = r23 * r23;
    r42 = r42 * r30;
    r41 = r41 + r42;
    r25 = fmaf(r5, r29, r25);
    r25 = fmaf(r26, r34, r25);
    r25 = fmaf(r27, r37, r25);
    r25 = fmaf(r9, r39, r25);
    r25 = fmaf(r8, r41, r25);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r41,
                                         r39);
    r37 = 9.99999999999999955e-07;
    r34 = r30 * r18;
    r23 = r23 * r21;
    r34 = fmaf(r24, r34, r23);
    r34 = fmaf(r7, r34, r6);
    r6 = r14 * r16;
    r6 = r6 * r19;
    r29 = fmaf(r15, r36, r6);
    r43 = r15 * r15;
    r43 = r30 * r43;
    r33 = r43 + r33;
    r44 = r14 * r17;
    r44 = fmaf(r19, r44, r35);
    r21 = fmaf(r24, r21, r38);
    r40 = r32 + r40;
    r38 = r18 * r18;
    r38 = r30 * r38;
    r40 = r40 + r38;
    r34 = fmaf(r5, r29, r34);
    r34 = fmaf(r27, r33, r34);
    r34 = fmaf(r26, r44, r34);
    r34 = fmaf(r8, r21, r34);
    r34 = fmaf(r9, r40, r34);
    r40 = copysign(1.0, r34);
    r40 = fmaf(r37, r40, r34);
    r37 = r40 * r40;
    r37 = 1.0 / r37;
    r42 = r32 + r42;
    r42 = r42 + r38;
    r42 = fmaf(r7, r42, r1);
    r13 = fmaf(r30, r13, r22);
    r22 = r19 * r18;
    r22 = fmaf(r24, r22, r23);
    r23 = r15 * r17;
    r23 = fmaf(r19, r23, r6);
    r36 = fmaf(r16, r36, r28);
    r31 = r32 + r31;
    r31 = r31 + r43;
    r42 = fmaf(r8, r13, r42);
    r42 = fmaf(r9, r22, r42);
    r42 = fmaf(r27, r23, r42);
    r42 = fmaf(r26, r36, r42);
    r42 = fmaf(r5, r31, r42);
    r31 = r42 * r42;
    r5 = r25 * r25;
    r5 = fmaf(r37, r5, r37 * r31);
    r5 = fmaf(r39, r5, r32);
    r5 = r41 * r5;
    r40 = 1.0 / r40;
    r5 = r5 * r40;
    r3 = fmaf(r25, r5, r3);
    r4 = fmaf(r2, r4, r0);
    r4 = fmaf(r42, r5, r4);
    r4 = fmaf(r4, r4, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointScore(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointFixedPointScoreKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_rTr,
              problem_size);
}

}  // namespace caspar