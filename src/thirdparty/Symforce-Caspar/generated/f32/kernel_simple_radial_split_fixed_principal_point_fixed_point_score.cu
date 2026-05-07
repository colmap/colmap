#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointScoreKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
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

  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;

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
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
    r10 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = fmaf(r12, r15, r13 * r18);
    r20 = r11 * r16;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r14, r17, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r13 * r15;
    r22 = fmaf(r4, r22, r12 * r18);
    r22 = fmaf(r14, r16, r22);
    r22 = fmaf(r11, r17, r22);
    r23 = r22 * r22;
    r23 = fmaf(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fmaf(r7, r24, r0);
    r0 = 2.00000000000000000e+00;
    r25 = fmaf(r14, r15, r11 * r18);
    r26 = r12 * r17;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r13, r16, r25);
    r26 = r0 * r25;
    r27 = r22 * r26;
    r28 = fmaf(r12, r16, r11 * r15);
    r28 = fmaf(r13, r17, r28);
    r28 = fmaf(r4, r28, r14 * r18);
    r14 = r10 * r28;
    r29 = fmaf(r19, r14, r27);
    r30 = r0 * r22;
    r31 = r19 * r26;
    r30 = fmaf(r28, r30, r31);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r32,
                       r33,
                       r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r15 * r17;
    r35 = r35 * r0;
    r36 = r16 * r18;
    r37 = fmaf(r0, r36, r35);
    r38 = r17 * r18;
    r39 = r15 * r16;
    r39 = r39 * r0;
    r38 = fmaf(r10, r38, r39);
    r40 = r16 * r16;
    r40 = r40 * r10;
    r41 = r21 + r40;
    r42 = r17 * r17;
    r42 = r10 * r42;
    r41 = r41 + r42;
    r24 = fmaf(r8, r29, r24);
    r24 = fmaf(r9, r30, r24);
    r24 = fmaf(r34, r37, r24);
    r24 = fmaf(r33, r38, r24);
    r24 = fmaf(r32, r41, r24);
  };
  LoadShared<2, float, float>(focal_and_distortion,
                              0 * focal_and_distortion_num_alloc,
                              focal_and_distortion_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_distortion_indices_loc[threadIdx.x].target,
                       r41,
                       r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = 9.99999999999999955e-07;
    r31 = fmaf(r22, r14, r31);
    r31 = fmaf(r7, r31, r6);
    r36 = fmaf(r10, r36, r35);
    r40 = r21 + r40;
    r35 = r15 * r15;
    r35 = r10 * r35;
    r40 = r40 + r35;
    r6 = r16 * r17;
    r6 = r6 * r0;
    r30 = r15 * r18;
    r30 = fmaf(r0, r30, r6);
    r29 = r0 * r19;
    r29 = r29 * r22;
    r26 = fmaf(r28, r26, r29);
    r43 = r25 * r25;
    r43 = r43 * r10;
    r23 = r43 + r23;
    r31 = fmaf(r32, r36, r31);
    r31 = fmaf(r34, r40, r31);
    r31 = fmaf(r33, r30, r31);
    r31 = fmaf(r8, r26, r31);
    r31 = fmaf(r9, r23, r31);
    r23 = copysign(1.0, r31);
    r23 = fmaf(r37, r23, r31);
    r37 = r23 * r23;
    r37 = 1.0 / r37;
    r31 = r24 * r24;
    r26 = r0 * r19;
    r26 = fmaf(r28, r26, r27);
    r26 = fmaf(r7, r26, r5);
    r7 = r17 * r18;
    r7 = fmaf(r0, r7, r39);
    r42 = r21 + r42;
    r42 = r42 + r35;
    r35 = r15 * r18;
    r35 = fmaf(r10, r35, r6);
    r14 = fmaf(r25, r14, r29);
    r20 = r21 + r20;
    r20 = r20 + r43;
    r26 = fmaf(r32, r7, r26);
    r26 = fmaf(r33, r42, r26);
    r26 = fmaf(r34, r35, r26);
    r26 = fmaf(r9, r14, r26);
    r26 = fmaf(r8, r20, r26);
    r20 = r26 * r26;
    r20 = fmaf(r37, r20, r37 * r31);
    r20 = fmaf(r38, r20, r21);
    r20 = r41 * r20;
    r23 = 1.0 / r23;
    r20 = r20 * r23;
    r2 = fmaf(r24, r20, r2);
    r4 = fmaf(r3, r4, r1);
    r4 = fmaf(r26, r20, r4);
    r4 = fmaf(r4, r4, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointScore(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointFixedPointScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar